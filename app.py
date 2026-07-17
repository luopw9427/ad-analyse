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
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

import r2_storage as r2
import db

ROOT = Path(__file__).resolve().parent
RUN_FROM_R2 = ROOT / "批处理" / "run_from_r2.py"
APP_TOKEN = os.environ.get("APP_TOKEN", "")

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


@app.get("/health")
def health():
    return {"ok": True, "service": "video-ad-analysis"}


@app.post("/analyze")
def analyze(req: AnalyzeReq, x_api_token: str = Header(default="")):
    _check_token(x_api_token)

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

    import time as _t
    _t0 = _t.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    log = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0
    db.record("analyze", req.project_id, req.product, ok,
              f"projects/{req.project_id}/",
              "" if ok else log[-500:], _t.time() - _t0)
    return {
        "ok": ok,
        "returncode": proc.returncode,
        "r2_prefix": f"projects/{req.project_id}/",
        "log_tail": log[-4000:],
    }


@app.post("/split")
def split(project_id: str = Form(...),
          ad_id: str = Form(default=""),
          precise: bool = Form(default=True),
          file: UploadFile = File(...),
          x_api_token: str = Header(default="")):
    """上传视频 → 云端拆分 → 切片+shots.json 写回 R2 projects/{project_id}/{ad_id}/。"""
    _check_token(x_api_token)

    import splitter
    import r2_storage as r2
    import traceback
    ad = ad_id or Path(file.filename or "ad").stem
    work = Path(tempfile.mkdtemp(prefix="split_"))
    try:
        video_path = work / (file.filename or "input.mp4")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        out_dir = work / ad
        shots = splitter.split_video(str(video_path), str(out_dir), precise=precise)

        # 回传切片 + shots.json 到 R2
        uploaded = 0
        for fp in sorted(out_dir.iterdir()):
            if fp.is_file():
                r2.upload_file(f"projects/{project_id}/{ad}/{fp.name}", str(fp))
                uploaded += 1

        db.record("split", project_id, ad, True,
                  f"projects/{project_id}/{ad}/", f"{len(shots)} shots", 0.0)
        return {
            "ok": True,
            "project_id": project_id,
            "ad_id": ad,
            "n_shots": len(shots),
            "uploaded": uploaded,
            "r2_prefix": f"projects/{project_id}/{ad}/",
            "shots": shots,
        }
    except Exception as e:
        db.record("split", project_id, ad, False, "", f"{type(e).__name__}: {e}", 0.0)
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-3000:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


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
