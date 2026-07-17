"""
视频广告分析 · HTTP 服务（部署到 Railway）
==========================================

对外只暴露一个触发接口，内部复用已跑通的 run_from_r2.py 逻辑：
    从 R2 拉切片 → 调 batch_analyze 分析 → 产物写回 R2。

接口：
    GET  /health                 健康检查
    POST /analyze                触发一次分析
      body: {"project_id":"test","product":"章鱼",
             "pipeline":"merged","pro":false,
             "overwrite":false,"cross_compare":false}
      header（可选）: X-API-Token: <APP_TOKEN>

环境变量（在 Railway Variables 里配）：
    R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY
    R2_BUCKET_NAME / R2_ENDPOINT / R2_PUBLIC_URL
    OPENROUTER_API_KEY
    APP_TOKEN（可选，设了就要求请求带同样的 X-API-Token）

本地起服务：
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
RUN_FROM_R2 = ROOT / "批处理" / "run_from_r2.py"
APP_TOKEN = os.environ.get("APP_TOKEN", "")

app = FastAPI(title="视频广告分析服务", version="0.1")


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
    if APP_TOKEN and x_api_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="invalid X-API-Token")

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

    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    log = (proc.stdout or "") + (proc.stderr or "")
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "r2_prefix": f"projects/{req.project_id}/",
        "log_tail": log[-4000:],
    }
