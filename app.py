"""
视频广告分析 · HTTP 服务（部署到 Railway）
==========================================

复用已跑通的 run_from_r2.py 逻辑：从 R2 拉切片 → 调 batch_analyze 分析 → 产物写回 R2。

接口：
    GET  /health                 健康检查
    POST /analyze                触发一次分析
      body: {"project_id":"test","product":"章鱼","pipeline":"merged",
             "pro":false,"overwrite":false,"cross_compare":false}
    GET  /result?project_id=test 查看某项目在 R2 的分析产物
      可选 ?include_content=false 只列文件不返回内容
    以上带保护的接口需请求头 X-API-Token（设了 APP_TOKEN 才校验）

环境变量（Railway Variables）：
    R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY
    R2_BUCKET_NAME / R2_ENDPOINT / R2_PUBLIC_URL
    OPENROUTER_API_KEY / APP_TOKEN（可选）
"""

from __future__ import annotations
import os
import sys
import subprocess
import tempfile
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

import r2_storage as r2
import db

ROOT = Path(__file__).resolve().parent
RUN_FROM_R2 = ROOT / "批处理" / "run_from_r2.py"
APP_TOKEN = os.environ.get("APP_TOKEN", "")
_SPLIT_LOCK = threading.Lock()   # 拆分串行，避免并发抢 CPU/内存

app = FastAPI(title="视频广告分析服务", version="0.3")


@app.on_event("startup")
def _startup():
    db.init_db()


def _check_token(x_api_token: str):
    if APP_TOKEN and x_api_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="invalid X-API-Token")


class AnalyzeReq(BaseModel):
    project_id: str
    product: str
    pipeline: Optional[str] = None   # merged / split
    pro: bool = False
    overwrite: bool = False
    cross_compare: bool = False
    hypotheses: Optional[str] = None   # 前置卖点提示（可选，逗号分隔）


@app.get("/")
def home():
    """结果查看网页（单页）。"""
    return FileResponse(ROOT / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "service": "video-ad-analysis"}


def _run_analyze_bg(req: "AnalyzeReq", job_id):
    """后台线程跑分析，完成后更新数据库任务状态。"""
    cmd = [sys.executable, str(RUN_FROM_R2),
           "--project", req.project_id, "--product", req.product]
    if req.pipeline:
        cmd += ["--pipeline", req.pipeline]
    if req.pro:
        cmd.append("--pro")
    if req.overwrite:
        cmd.append("--overwrite")
    if req.cross_compare:
        cmd.append("--cross-compare")
    if req.hypotheses:
        cmd += ["--hypotheses", req.hypotheses]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        log = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0
        db.finish_job(job_id, ok, f"projects/{req.project_id}/",
                      "" if ok else log[-800:], time.time() - t0)
    except Exception as e:
        db.finish_job(job_id, False, "", f"{type(e).__name__}: {e}", time.time() - t0)


@app.post("/analyze")
def analyze(req: AnalyzeReq, x_api_token: str = Header(default="")):
    """异步触发分析：立即返回 job_id，后台跑，前端轮询 /jobs 看进度。"""
    _check_token(x_api_token)
    job_id = db.start_job("analyze", req.project_id, req.product)
    threading.Thread(target=_run_analyze_bg, args=(req, job_id), daemon=True).start()
    return {
        "ok": True,
        "started": True,
        "job_id": job_id,
        "r2_prefix": f"projects/{req.project_id}/",
        "note": "分析已在后台开始，查任务记录看进度",
    }


def _run_split_bg(project_id, ad, video_path_str, work_str, precise, job_id):
    """后台线程：拆分 → 切片写回 R2 → 更新任务状态。"""
    import splitter
    import r2_storage as r2
    t0 = time.time()
    try:
        with _SPLIT_LOCK:      # 同一时刻只跑一个拆分
            out_dir = Path(work_str) / ad
            shots = splitter.split_video(video_path_str, str(out_dir), precise=precise)
        for fp in sorted(out_dir.iterdir()):
            if fp.is_file():
                r2.upload_file(f"projects/{project_id}/{ad}/{fp.name}", str(fp))
        db.finish_job(job_id, True, f"projects/{project_id}/{ad}/",
                      f"{len(shots)} shots", time.time() - t0)
    except Exception as e:
        db.finish_job(job_id, False, "", f"{type(e).__name__}: {e}", time.time() - t0)
    finally:
        shutil.rmtree(work_str, ignore_errors=True)


@app.post("/split")
def split(project_id: str = Form(...),
          ad_id: str = Form(default=""),
          precise: bool = Form(default=True),
          file: UploadFile = File(...),
          x_api_token: str = Header(default="")):
    """异步：先存上传文件，立即返回 job_id，后台拆分+回传，前端轮询 /jobs。"""
    _check_token(x_api_token)
    ad = ad_id or Path(file.filename or "ad").stem
    work = Path(tempfile.mkdtemp(prefix="split_"))
    video_path = work / (file.filename or "input.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job_id = db.start_job("split", project_id, ad)
    threading.Thread(target=_run_split_bg,
                     args=(project_id, ad, str(video_path), str(work), precise, job_id),
                     daemon=True).start()
    return {"ok": True, "started": True, "job_id": job_id, "ad_id": ad}


@app.get("/result")
def result(project_id: str, include_content: bool = True,
           x_api_token: str = Header(default="")):
    _check_token(x_api_token)

    prefix = f"projects/{project_id}/"
    keys = r2.list_keys(prefix)
    if not keys:
        raise HTTPException(status_code=404, detail=f"{prefix} 下没有任何产物")

    objects = [{"key": k, "url": r2.public_url(k)} for k in keys]

    reports = {}
    if include_content:
        for k in keys:
            if k.endswith(".json"):
                try:
                    reports[k] = r2.read_json(k)
                except Exception as e:
                    reports[k] = {"_error": str(e)}

    return {
        "project_id": project_id,
        "count": len(keys),
        "objects": objects,
        "reports": reports,
    }


@app.get("/jobs")
def jobs(limit: int = 50, x_api_token: str = Header(default="")):
    _check_token(x_api_token)
    return {"jobs": db.recent(limit), "db_enabled": db.enabled()}


class DeleteReq(BaseModel):
    project_id: str
    ad_id: Optional[str] = None        # 指定则只删该广告
    analysis_only: bool = False        # 只删分析产物，保留切片
    clear_jobs: bool = False           # 同时清理该项目的任务记录


@app.post("/delete")
def delete(req: DeleteReq, x_api_token: str = Header(default="")):
    """删除 R2 上的产物，释放存储。"""
    _check_token(x_api_token)
    if not req.project_id:
        raise HTTPException(status_code=400, detail="缺少 project_id")

    if req.analysis_only:
        # 只删分析产物：各广告的 analysis_output/ + 项目级汇总文件
        prefix = f"projects/{req.project_id}/"
        keys = [k for k in r2.list_keys(prefix)
                if "/analysis_output" in k or k.endswith("共性镜头prompt.json")
                or k.endswith("横向对比.md")]
        if req.ad_id:
            keys = [k for k in keys if k.startswith(f"{prefix}{req.ad_id}/")]
        deleted = r2.delete_keys(keys)
        scope = f"{req.project_id}/{req.ad_id or ''} 的分析产物"
    else:
        prefix = (f"projects/{req.project_id}/{req.ad_id}/"
                  if req.ad_id else f"projects/{req.project_id}/")
        deleted = r2.delete_prefix(prefix)
        scope = prefix

    jobs_deleted = db.delete_jobs(req.project_id) if req.clear_jobs else 0
    return {"ok": True, "deleted": deleted, "scope": scope, "jobs_deleted": jobs_deleted}
