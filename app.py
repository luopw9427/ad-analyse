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
MODAL_SPLIT_URL = os.environ.get("MODAL_SPLIT_URL", "")     # Modal GPU 拆分接口
MODAL_SPLIT_TOKEN = os.environ.get("MODAL_SPLIT_TOKEN", "")
# ── 拆分成本模型 ────────────────────────────────────────────────
# Modal 按容器存活时长 × 资源单价计费（官方价：T4 $0.59/h、CPU $0.0473/核/h、内存 $0.0080/GiB/h）。
# 容器规格固定为 T4 + 1核 + 2GiB，故单价确定：
MODAL_RATE_USD_PER_SEC = float(os.environ.get("MODAL_RATE_USD_PER_SEC", "0.00018147"))
# 关键：冷启动（拉镜像→装载CUDA→模型上GPU）这段发生在容器内计时之前，测不到，只能建模。
# 由三次实测反推：计费时长 165/275/220 秒，而执行仅 19~28 秒 → 冷启动约 180 秒。
# 想更准就按 Modal 后台 Usage 的实际增量调这个值（不用改代码，Railway 变量即可）。
MODAL_COLD_START_SEC = float(os.environ.get("MODAL_COLD_START_SEC", "180"))
MODAL_COST_MULTIPLIER = float(os.environ.get("MODAL_COST_MULTIPLIER", "0.25"))  # 实测校准：估算约为真实4倍

import re as _re
_COST_RE = _re.compile(r"cost=\$([0-9.]+)")
_TOK_RE = _re.compile(r"usage:\s*prompt=(\d+)[^\n]*?completion=(\d+)")


def _sum_cost_from_log(log: str):
    """从分析日志汇总 OpenRouter 返回的真实费用与 token 数。"""
    cost = sum(float(m.group(1)) for m in _COST_RE.finditer(log or ""))
    tin = tout = 0
    for m in _TOK_RE.finditer(log or ""):
        tin += int(m.group(1)); tout += int(m.group(2))
    return cost, tin, tout
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
    openrouter_key: Optional[str] = None   # 前端传入，留空则用服务端环境变量


@app.get("/")
def home():
    """结果查看网页（单页）。禁用缓存，避免改版后浏览器仍显示旧页面。"""
    return FileResponse(
        ROOT / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache", "Expires": "0"},
    )


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
        env = os.environ.copy()
        if req.openrouter_key:
            env["OPENROUTER_API_KEY"] = req.openrouter_key
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
        log = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0
        cost, tin, tout = _sum_cost_from_log(log)
        db.finish_job(job_id, ok, f"projects/{req.project_id}/",
                      (f"LLM 调用 {tin}+{tout} tokens" if ok else log[-800:]),
                      time.time() - t0,
                      cost_usd=cost, tokens_in=tin, tokens_out=tout, cost_exact=True)
    except Exception as e:
        db.finish_job(job_id, False, "", f"{type(e).__name__}: {e}", time.time() - t0)


@app.post("/analyze")
def analyze(req: AnalyzeReq, x_api_token: str = Header(default="")):
    """异步触发分析：立即返回 job_id，后台跑，前端轮询 /jobs 看进度。"""
    _check_token(x_api_token)
    job_id = db.start_job("analyze", req.project_id, "", product=req.product)
    threading.Thread(target=_run_analyze_bg, args=(req, job_id), daemon=True).start()
    return {
        "ok": True,
        "started": True,
        "job_id": job_id,
        "r2_prefix": f"projects/{req.project_id}/",
        "note": "分析已在后台开始，查任务记录看进度",
    }


def _split_via_gpu(project_id, ad, video_path_str, precise, keep_raw):
    """GPU 路径：原视频存 R2 → 调 Modal → Modal 直接把切片写回 R2。"""
    import requests
    import r2_storage as r2
    if not MODAL_SPLIT_URL or not MODAL_SPLIT_TOKEN:
        raise RuntimeError("未配置 MODAL_SPLIT_URL / MODAL_SPLIT_TOKEN")

    raw_key = f"raw/{project_id}/{ad}.mp4"
    r2.upload_file(raw_key, video_path_str)
    try:
        resp = requests.post(
            MODAL_SPLIT_URL,
            json={"token": MODAL_SPLIT_TOKEN, "project_id": project_id,
                  "ad_id": ad, "raw_key": raw_key, "precise": precise},
            timeout=1800,
        )
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"调用 Modal 失败: {e}")
    if not data.get("ok"):
        raise RuntimeError(f"Modal 拆分失败: {data.get('error') or resp.status_code}")

    if not keep_raw:
        try:
            r2.delete_keys([raw_key])
        except Exception as e:
            print(f"[split] 清理原视频失败（不影响结果）: {e}")
    return data


def _run_split_bg(project_id, ad, video_path_str, work_str, precise, job_id,
                  engine="cpu", keep_raw=False):
    """后台线程：按引擎拆分 → 切片写回 R2 → 更新任务状态。"""
    import r2_storage as r2
    t0 = time.time()
    try:
        if engine == "gpu":
            data = _split_via_gpu(project_id, ad, video_path_str, precise, keep_raw)
            n = data.get("n_shots", 0)
            elapsed = float(data.get("elapsed") or 0)
            billed = float(data.get("billed_seconds") or 0)
            # 成本 = (容器内实测计费秒数 + 冷启动开销) × 单价
            cold = bool(data.get("cold_start"))
            billed_total = billed + (MODAL_COLD_START_SEC if cold else 0.0)
            cost = billed_total * MODAL_RATE_USD_PER_SEC * MODAL_COST_MULTIPLIER
            db.finish_job(job_id, True, data.get("r2_prefix", ""),
                          f"{n} shots · GPU {'冷启动' if cold else '热容器'} · "
                          f"执行{elapsed}s / 计费{billed_total:.0f}s",
                          time.time() - t0,
                          cost_usd=cost, cost_exact=False)
        else:
            import splitter
            with _SPLIT_LOCK:      # CPU 拆分串行，避免抢资源
                out_dir = Path(work_str) / ad
                shots = splitter.split_video(video_path_str, str(out_dir), precise=precise)
            for fp in sorted(out_dir.iterdir()):
                if fp.is_file():
                    r2.upload_file(f"projects/{project_id}/{ad}/{fp.name}", str(fp))
            db.finish_job(job_id, True, f"projects/{project_id}/{ad}/",
                          f"{len(shots)} shots (CPU·含Railway订阅)", time.time() - t0,
                          cost_usd=0.0, cost_exact=True)
    except Exception as e:
        db.finish_job(job_id, False, "", f"{type(e).__name__}: {e}", time.time() - t0)
    finally:
        shutil.rmtree(work_str, ignore_errors=True)


@app.post("/split")
def split(project_id: str = Form(...),
          ad_id: str = Form(default=""),
          precise: bool = Form(default=True),
          engine: str = Form(default="cpu"),        # cpu / gpu
          keep_raw: bool = Form(default=False),     # GPU 路径是否保留原视频
          file: UploadFile = File(...),
          x_api_token: str = Header(default="")):
    """异步：先存上传文件，立即返回 job_id，后台拆分+回传，前端轮询 /jobs。"""
    _check_token(x_api_token)
    ad = ad_id or Path(file.filename or "ad").stem
    work = Path(tempfile.mkdtemp(prefix="split_"))
    video_path = work / (file.filename or "input.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    engine = (engine or "cpu").lower()
    job_id = db.start_job("split", project_id, ad, engine=engine)
    threading.Thread(target=_run_split_bg,
                     args=(project_id, ad, str(video_path), str(work), precise,
                           job_id, engine, keep_raw),
                     daemon=True).start()
    return {"ok": True, "started": True, "job_id": job_id, "ad_id": ad, "engine": engine}


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


class JobDeleteReq(BaseModel):
    job_id: int


@app.post("/jobs/delete")
def jobs_delete(req: JobDeleteReq, x_api_token: str = Header(default="")):
    """删除单条任务记录（不动 R2 文件）。"""
    _check_token(x_api_token)
    return {"ok": True, "deleted": db.delete_job(req.job_id)}
