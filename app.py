"""
视频广告分析 · HTTP 服务（部署到 Railway）
==========================================

复用已跑通的 run_from_r2.py 逻辑：从 R2 拉切片 → 调 batch_analyze 分析 → 产物写回 R2。

接口：
    GET  /health                 健康检查
    POST /analyze                触发一次分析
      body: {"project_id":"test","product":"章鱼","pipeline":"merged",
             "pro":false,"overwrite":false,"cross_compare":false}
    GET  /result?project_id=test 查看某分组在 R2 的分析产物
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
import json
import tempfile
import shutil
import threading
import time
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
import zipfile
from pydantic import BaseModel

import r2_storage as r2
import db
import translator
import reviser

ROOT = Path(__file__).resolve().parent
RUN_FROM_R2 = ROOT / "批处理" / "run_from_r2.py"
APP_TOKEN = os.environ.get("APP_TOKEN", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
FAV_DEFAULT_USER = os.environ.get("FAV_DEFAULT_USER", "default")
# ── 登录 / 鉴权 ────────────────────────────────────────────────
AUTH_SECRET = os.environ.get("AUTH_SECRET", "") or APP_TOKEN or "change-me-please"
ADMIN_USER = os.environ.get("ADMIN_USER", "")
ADMIN_PIN = os.environ.get("ADMIN_PIN", "")
TOKEN_DAYS_REMEMBER = int(os.environ.get("TOKEN_DAYS_REMEMBER", "0"))   # 0 = 记住则永久
TOKEN_HOURS_SESSION = int(os.environ.get("TOKEN_HOURS_SESSION", "12"))  # 不记住的短效
# 分析产物里需要翻译成英文的 JSON（键名结构保留，只译值）→ 存为同名 .en.json
TRANSLATE_NAMES = ("ad_aggregate.json", "video_prompt.json",
                   "shot_prompts.json", "共性镜头prompt.json")
TRANSLATE_WORKERS = int(os.environ.get("TRANSLATE_WORKERS", "4"))
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
    if ADMIN_USER and ADMIN_PIN:
        db.user_ensure_admin(ADMIN_USER, ADMIN_PIN)          # 预置管理员（方案 A）
    elif db.admin_count() == 0:
        db.fav_user_ensure(FAV_DEFAULT_USER, "", role="admin")  # 无管理员则把 default 设为管理员，供首次进入建号
        print("[auth] 未配置 ADMIN_USER/ADMIN_PIN，已把 default（空密码）设为临时管理员，请尽快登录并创建正式管理员")


# ── 签名 token（无状态，HMAC-SHA256）──────────────────────────────
import base64 as _b64, hmac as _hmac, hashlib as _hl

def _b64u(b: bytes) -> str:
    return _b64.urlsafe_b64encode(b).decode().rstrip("=")

def _b64u_dec(s: str) -> bytes:
    return _b64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

def make_token(username: str, role: str, pin_ver: int, remember: bool) -> str:
    now = int(time.time())
    if remember:
        exp = 0 if TOKEN_DAYS_REMEMBER <= 0 else now + TOKEN_DAYS_REMEMBER * 86400  # 0=永久
    else:
        exp = now + TOKEN_HOURS_SESSION * 3600
    payload = {"u": username, "r": role, "v": int(pin_ver), "iat": now, "exp": exp}
    body = _b64u(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64u(_hmac.new(AUTH_SECRET.encode(), body.encode(), _hl.sha256).digest())
    return body + "." + sig

def parse_token(tok: str):
    try:
        body, sig = tok.split(".")
        good = _b64u(_hmac.new(AUTH_SECRET.encode(), body.encode(), _hl.sha256).digest())
        if not _hmac.compare_digest(sig, good):
            return None
        p = json.loads(_b64u_dec(body))
        if p.get("exp") and int(p["exp"]) > 0 and int(p["exp"]) < time.time():
            return None
        return p
    except Exception:
        return None


def current_user(authorization: str = Header(default="")):
    """从 Authorization: Bearer <token> 解析当前登录用户；无效则 401。"""
    tok = authorization.split(" ", 1)[1] if authorization.lower().startswith("bearer ") else authorization
    p = parse_token(tok)
    if not p:
        raise HTTPException(status_code=401, detail="未登录或登录已失效")
    u = db.user_get(p.get("u", ""))
    if not u or u.get("disabled"):
        raise HTTPException(status_code=401, detail="账号不可用")
    if int(u.get("pin_ver", 1)) != int(p.get("v", 1)):
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    return {"username": u["username"], "role": u["role"]}


def require_admin(user: dict = Depends(current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


def _client_ip(request: Request) -> str:
    """取真实客户端 IP（Railway 等反代下用 X-Forwarded-For 首个）。"""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else ""


def _epid(owner: str, project: str) -> str:
    """R2 命名空间：按用户隔离分组，故 R2 里的"分组标识" = 用户~分组名。
    这样不同用户可建同名分组、互不覆盖；DB 里仍分别存 owner + project(分组名)。"""
    owner = (owner or "").strip() or "default"
    project = (project or "").replace("~", "-").replace("/", "-")
    return f"{owner}~{project}"


class AnalyzeReq(BaseModel):
    project_id: str
    product: str
    pipeline: Optional[str] = None   # merged / split
    pro: bool = False
    overwrite: bool = False
    cross_compare: bool = False
    hypotheses: Optional[str] = None   # 前置卖点提示（可选，逗号分隔）
    openrouter_key: Optional[str] = None   # 前端传入，留空则用服务端环境变量
    deepseek_key: Optional[str] = None     # 翻译用；留空则用服务端 DEEPSEEK_API_KEY
    translate: bool = True                 # 分析完成后是否自动翻译成英文
    batch: Optional[str] = None            # 本次批次；给了则只分析这一批
    custom_prompts: Optional[dict] = None  # {环节key: 提示词文本}，覆盖默认


@app.get("/")
def home():
    """结果查看网页（单页）。禁用缓存，避免改版后浏览器仍显示旧页面。"""
    return FileResponse(
        ROOT / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache", "Expires": "0"},
    )


@app.get("/prompts")
def prompts_page():
    """提示词编辑页（单页）。"""
    return FileResponse(ROOT / "prompts.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/account")
def account_page():
    """个人中心 / 账号设置（独立页）。"""
    return FileResponse(ROOT / "account.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/admin")
def admin_page():
    """管理员后台（独立页）。"""
    return FileResponse(ROOT / "admin.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/prompts/defaults")
def prompts_defaults():
    """返回 10 个环节的默认提示词 + 必需占位符（供编辑页展示/校验）。"""
    try:
        with open(ROOT / "prompts_defaults.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取默认提示词失败: {e}")


@app.get("/health")
def health():
    return {"ok": True, "service": "video-ad-analysis"}


def _run_analyze_bg(req: "AnalyzeReq", job_id, owner=""):
    """后台线程跑分析，完成后更新数据库任务状态。"""
    epid = _epid(owner, req.project_id)   # R2 命名空间（按用户隔离）
    cmd = [sys.executable, str(RUN_FROM_R2),
           "--project", epid, "--product", req.product]
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
    if req.batch:
        cmd += ["--batch", req.batch]
    _prefix = f"projects/{epid}/" + (f"{req.batch}/" if req.batch else "")

    t0 = time.time()
    try:
        env = os.environ.copy()
        if req.openrouter_key:
            env["OPENROUTER_API_KEY"] = req.openrouter_key
        _pf = None
        if req.custom_prompts:
            _pf = tempfile.mktemp(prefix="prompts_", suffix=".json")
            with open(_pf, "w", encoding="utf-8") as _f:
                json.dump(req.custom_prompts, _f, ensure_ascii=False)
            env["PROMPTS_FILE"] = _pf
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
        finally:
            if _pf:
                try: os.remove(_pf)
                except Exception: pass
        log = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0
        cost, tin, tout = _sum_cost_from_log(log)
        # 提取每条广告的成败摘要（batch_analyze 打印 "成功: X / 总计: Y" 及失败明细）
        import re as _re2
        m = _re2.search(r"成功:\s*(\d+)\s*/\s*总计:\s*(\d+)", log)
        fails = _re2.findall(r"✗\s*([^\n:]+):\s*([^\n]+)", log)
        summ = (f"{m.group(1)}/{m.group(2)} 广告成功" if m else f"LLM {tin}+{tout} tokens")
        if fails:
            summ += " · 失败: " + "; ".join(f"{a.strip()}({b.strip()[:40]})" for a, b in fails[:5])
        db.finish_job(job_id, ok, _prefix,
                      (summ if ok else log[-800:]),
                      time.time() - t0,
                      cost_usd=cost, tokens_in=tin, tokens_out=tout, cost_exact=True)
        # 分析成功后：另起后台线程做中→英翻译，结果先展示、英文陆续填充
        if ok and req.translate:
            ds_key = req.deepseek_key or DEEPSEEK_API_KEY
            if ds_key:
                instr = translator.pick_instr((req.custom_prompts or {}).get("translate"))
                threading.Thread(
                    target=_translate_bg,
                    args=(req.project_id, req.batch or "", instr, ds_key, owner),
                    daemon=True).start()
            else:
                print("[translate] 跳过：未配置 DEEPSEEK_API_KEY")
    except Exception as e:
        db.finish_job(job_id, False, "", f"{type(e).__name__}: {e}", time.time() - t0)


def _translate_bg(project_id, batch, instr, ds_key, owner=""):
    """后台并发翻译：把批次内生成的中文 JSON 译成英文，逐个上传 <名>.en.json。"""
    epid = _epid(owner, project_id)
    prefix = f"projects/{epid}/" + (f"{batch}/" if batch else "")
    ref = batch or project_id
    job_id = db.start_job("translate", project_id, ref, batch=batch or "", owner=owner)
    t0 = time.time()
    try:
        keys = r2.list_keys(prefix)
        # 选出需翻译且尚未翻译（跳过 .en.json 自身）的中文 JSON
        todo = [k for k in keys
                if any(k.endswith(n) for n in TRANSLATE_NAMES) and not k.endswith(".en.json")]
        if not todo:
            db.finish_job(job_id, True, prefix, "无可翻译产物", time.time() - t0,
                          cost_usd=0.0, cost_exact=False)
            return

        client = translator._client(ds_key)
        n_ok = n_fail = 0
        tot_in = tot_out = 0

        def _one(k):
            try:
                obj = r2.read_json(k)
            except Exception as e:
                return k, False, 0, 0, f"读取失败 {e}"
            en, tin, tout = translator.translate_obj(obj, instr, client)
            if en is None:
                return k, False, tin, tout, "翻译失败"
            en_key = k[:-len(".json")] + ".en.json"   # 同路径 .en.json
            try:
                r2.upload_json(en_key, en)
            except Exception as e:
                return k, False, tin, tout, f"上传失败 {e}"
            return k, True, tin, tout, en_key

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=TRANSLATE_WORKERS) as ex:
            futs = [ex.submit(_one, k) for k in todo]
            for f in as_completed(futs):
                _k, good, tin, tout, info = f.result()
                tot_in += tin; tot_out += tout
                if good:
                    n_ok += 1; print(f"[translate] ✓ {info}")
                else:
                    n_fail += 1; print(f"[translate] ✗ {_k}: {info}")

        cost = translator.cost_of(tot_in, tot_out)
        summ = f"英文翻译 {n_ok}/{len(todo)} 完成" + (f" · 失败{n_fail}" if n_fail else "")
        db.finish_job(job_id, n_fail == 0, prefix, summ, time.time() - t0,
                      cost_usd=cost, tokens_in=tot_in, tokens_out=tot_out, cost_exact=False)
    except Exception as e:
        db.finish_job(job_id, False, "", f"翻译异常 {type(e).__name__}: {e}", time.time() - t0)


@app.post("/analyze")
def analyze(req: AnalyzeReq, user: dict = Depends(current_user)):
    """异步触发分析：立即返回 job_id，后台跑，前端轮询 /jobs 看进度。"""
    # 对象名用该批次的视频名（和拆分一致），而非批次 id
    epid = _epid(user["username"], req.project_id)
    ref = req.batch or ""
    try:
        if req.batch:
            pfx = f"projects/{epid}/{req.batch}/"
            ads = sorted({k[len(pfx):].split("/")[0] for k in r2.list_keys(pfx)
                          if "/" in k[len(pfx):]})
            if ads:
                ref = "、".join(ads)
    except Exception:
        pass
    job_id = db.start_job("analyze", req.project_id, ref, product=req.product, batch=req.batch or "", owner=user["username"])
    threading.Thread(target=_run_analyze_bg, args=(req, job_id, user["username"]), daemon=True).start()
    return {
        "ok": True, "started": True, "job_id": job_id,
        "r2_prefix": f"projects/{epid}/" + (f"{req.batch}/" if req.batch else ""),
        "note": "分析已在后台开始，查任务记录看进度",
    }


def _split_via_gpu(project_id, batch, ad, video_path_str, precise, keep_raw):
    """GPU 路径：原视频存 R2 → 调 Modal → Modal 直接把切片写回 R2。"""
    import requests
    import r2_storage as r2
    if not MODAL_SPLIT_URL or not MODAL_SPLIT_TOKEN:
        raise RuntimeError("未配置 MODAL_SPLIT_URL / MODAL_SPLIT_TOKEN")

    raw_key = f"raw/{project_id}/{batch}/{ad}.mp4"
    out_prefix = f"projects/{project_id}/{batch}/{ad}/"
    r2.upload_file(raw_key, video_path_str)
    try:
        resp = requests.post(
            MODAL_SPLIT_URL,
            json={"token": MODAL_SPLIT_TOKEN, "project_id": project_id,
                  "ad_id": ad, "raw_key": raw_key, "out_prefix": out_prefix,
                  "precise": precise},
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


def _run_split_bg(project_id, batch, ad, video_path_str, work_str, precise, job_id,
                  engine="cpu", keep_raw=False):
    """后台线程：按引擎拆分 → 切片写回 R2 → 更新任务状态。"""
    import r2_storage as r2
    t0 = time.time()
    try:
        if engine == "gpu":
            data = _split_via_gpu(project_id, batch, ad, video_path_str, precise, keep_raw)
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
                    r2.upload_file(f"projects/{project_id}/{batch}/{ad}/{fp.name}", str(fp))
            db.finish_job(job_id, True, f"projects/{project_id}/{batch}/{ad}/",
                          f"{len(shots)} shots (CPU·含Railway订阅)", time.time() - t0,
                          cost_usd=0.0, cost_exact=True)
    except Exception as e:
        db.finish_job(job_id, False, "", f"{type(e).__name__}: {e}", time.time() - t0)
    finally:
        shutil.rmtree(work_str, ignore_errors=True)


def _split_one(project_id, batch, ad, video_path_str, work_str, engine, precise, keep_raw):
    """拆一个视频，返回 (镜头数, 成本)。"""
    import r2_storage as r2
    if engine == "gpu":
        data = _split_via_gpu(project_id, batch, ad, video_path_str, precise, keep_raw)
        n = data.get("n_shots", 0)
        billed = float(data.get("billed_seconds") or 0)
        cold = bool(data.get("cold_start"))
        billed_total = billed + (MODAL_COLD_START_SEC if cold else 0.0)
        return n, billed_total * MODAL_RATE_USD_PER_SEC * MODAL_COST_MULTIPLIER
    import splitter
    with _SPLIT_LOCK:
        out_dir = Path(work_str) / ad
        shots = splitter.split_video(video_path_str, str(out_dir), precise=precise)
    for fp in sorted(out_dir.iterdir()):
        if fp.is_file():
            r2.upload_file(f"projects/{project_id}/{batch}/{ad}/{fp.name}", str(fp))
    return len(shots), 0.0


def _download_url(url, dest):
    import requests
    with requests.get(url, stream=True, timeout=600,
                      headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                if chunk:
                    f.write(chunk)
    return dest


def _run_split_batch_bg(project_id, batch, items, work_str, engine, precise, keep_raw, job_id):
    """一个批次一个后台任务：依次处理每个视频（本地文件或 URL 下载），汇总成败/成本。"""
    import r2_storage as r2
    t0 = time.time(); okn = 0; total_shots = 0; total_cost = 0.0; fails = []
    for it in items:
        ad = it["ad"]; src = it.get("src"); path = it.get("path")
        try:
            if not path:                       # URL 项：先下载
                path = os.path.join(work_str, ad + ".mp4")
                _download_url(src, path)
            n, cost = _split_one(project_id, batch, ad, path, work_str, engine, precise, keep_raw)
            if src:                            # 记录原视频 URL，供结果页展示可跳转
                sj = os.path.join(work_str, ad + "_source.json")
                with open(sj, "w", encoding="utf-8") as f:
                    json.dump({"url": src}, f, ensure_ascii=False)
                r2.upload_file(f"projects/{project_id}/{batch}/{ad}/source.json", sj)
            okn += 1; total_shots += n; total_cost += cost
        except Exception as e:
            fails.append(f"{ad}({type(e).__name__})")
    detail = f"{okn}/{len(items)} 视频 · {total_shots} 镜头"
    if fails:
        detail += " · 失败: " + "; ".join(fails[:5])
    db.finish_job(job_id, len(fails) == 0, f"projects/{project_id}/{batch}/",
                  detail, time.time() - t0, cost_usd=total_cost, cost_exact=(engine != "gpu"))
    shutil.rmtree(work_str, ignore_errors=True)


def _ad_from_url(url, used):
    from urllib.parse import urlparse, unquote
    import re as _re
    base = unquote(urlparse(url).path.rsplit("/", 1)[-1])
    base = _re.sub(r"\.[a-zA-Z0-9]{1,5}$", "", base)          # 去扩展名
    base = _re.sub(r"[^\w一-龥-]", "", base) or "url"  # 只留字母数字中文-_
    ad = base
    i = 2
    while ad in used:
        ad = f"{base}_{i}"; i += 1
    used.add(ad)
    return ad


@app.post("/split_batch")
def split_batch(project_id: str = Form(...), batch: str = Form(default=""),
                engine: str = Form(default="cpu"), precise: bool = Form(default=True),
                keep_raw: bool = Form(default=False), urls: str = Form(default=""),
                files: Optional[List[UploadFile]] = File(None),
                user: dict = Depends(current_user)):
    """本次上传（本地文件 + 粘贴的 URL）= 一个批次 = 一条任务。"""
    batch = batch or time.strftime("%Y%m%d-%H%M%S")
    engine = (engine or "cpu").lower()
    work = Path(tempfile.mkdtemp(prefix="splitb_"))
    items = []; used = set()
    for f in (files or []):
        ad = Path(f.filename or "ad").stem
        used.add(ad)
        vp = work / (f.filename or (ad + ".mp4"))
        with open(vp, "wb") as out:
            shutil.copyfileobj(f.file, out)
        items.append({"ad": ad, "path": str(vp)})
    try:
        url_list = json.loads(urls) if urls else []
    except Exception:
        url_list = []
    for u in url_list:
        u = (u or "").strip()
        if not u:
            continue
        items.append({"ad": _ad_from_url(u, used), "src": u})
    if not items:
        raise HTTPException(status_code=400, detail="没有视频（本地文件或 URL 至少一个）")
    ref = "、".join(it["ad"] for it in items)[:60]
    job_id = db.start_job("split", project_id, ref, engine=engine, batch=batch, owner=user["username"])
    epid = _epid(user["username"], project_id)   # R2 命名空间；db 里仍存分组名
    threading.Thread(target=_run_split_batch_bg,
                     args=(epid, batch, items, str(work), engine, precise, keep_raw, job_id),
                     daemon=True).start()
    return {"ok": True, "started": True, "job_id": job_id, "batch": batch, "n": len(items)}


@app.post("/split")
def split(project_id: str = Form(...),
          ad_id: str = Form(default=""),
          precise: bool = Form(default=True),
          engine: str = Form(default="cpu"),        # cpu / gpu
          keep_raw: bool = Form(default=False),     # GPU 路径是否保留原视频
          batch: str = Form(default=""),            # 本次上传批次 id
          file: UploadFile = File(...),
          user: dict = Depends(current_user)):
    """异步：先存上传文件，立即返回 job_id，后台拆分+回传，前端轮询 /jobs。"""
    ad = ad_id or Path(file.filename or "ad").stem
    work = Path(tempfile.mkdtemp(prefix="split_"))
    video_path = work / (file.filename or "input.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    engine = (engine or "cpu").lower()
    batch = batch or time.strftime("%Y%m%d-%H%M%S")
    job_id = db.start_job("split", project_id, ad, engine=engine, batch=batch, owner=user["username"])
    epid = _epid(user["username"], project_id)
    threading.Thread(target=_run_split_bg,
                     args=(epid, batch, ad, str(video_path), str(work), precise,
                           job_id, engine, keep_raw),
                     daemon=True).start()
    return {"ok": True, "started": True, "job_id": job_id, "ad_id": ad, "engine": engine, "batch": batch}


@app.get("/result")
def result(project_id: str, include_content: bool = True, batch: str = "", owner: str = "",
           user: dict = Depends(current_user)):
    # owner 缺省=当前用户；给了别人则查看别人的分组（团队可见）
    owner = (owner or "").strip() or user["username"]
    epid = _epid(owner, project_id)
    real_pfx = f"projects/{epid}/"                   # R2 真实前缀（带用户命名空间）
    show_pfx = f"projects/{project_id}/"             # 返回给前端时用的前缀（隐去用户）
    prefix = real_pfx + (f"{batch}/" if batch else "")
    keys = r2.list_keys(prefix)
    if not keys:
        raise HTTPException(status_code=404, detail=f"该分组下没有任何产物")

    def _show(k):   # 把真实 key 的用户命名空间隐去，前端仍按 projects/{分组}/ 解析
        return show_pfx + k[len(real_pfx):] if k.startswith(real_pfx) else k

    objects = [{"key": _show(k), "url": r2.public_url(k)} for k in keys]

    WANT = ("ad_aggregate.json", "video_prompt.json", "shot_prompts.json", "共性镜头prompt.json", "source.json",
            "ad_aggregate.en.json", "video_prompt.en.json", "shot_prompts.en.json", "共性镜头prompt.en.json")
    reports = {}
    if include_content:
        for k in keys:
            if any(k.endswith(w) for w in WANT):
                try:
                    reports[_show(k)] = r2.read_json(k)
                except Exception as e:
                    reports[_show(k)] = {"_error": str(e)}

    return {
        "project_id": project_id,
        "owner": owner,
        "count": len(keys),
        "objects": objects,
        "reports": reports,
    }


@app.get("/jobs")
def jobs(limit: int = 50, user: dict = Depends(current_user)):
    return {"jobs": db.recent(limit), "db_enabled": db.enabled()}


class DeleteReq(BaseModel):
    project_id: str
    ad_id: Optional[str] = None        # 指定则只删该广告
    batch: Optional[str] = None        # 指定批次；配合 ad_id 删批次内单条，单独给删整批
    analysis_only: bool = False        # 只删分析产物，保留切片
    clear_jobs: bool = False           # 同时清理该分组的任务记录
    owner: Optional[str] = None        # 归属用户；缺省=当前用户（删别人的需管理员）


@app.post("/delete")
def delete(req: DeleteReq, request: Request, user: dict = Depends(current_user)):
    """删除 R2 上的产物，释放存储。权限：管理员随意；普通用户仅限自己的分组；删整个分组仅管理员。"""
    if not req.project_id:
        raise HTTPException(status_code=400, detail="缺少 project_id")

    is_admin = user["role"] == "admin"
    target_owner = (req.owner or "").strip() or user["username"]
    # 分组已按用户隔离：本人可删自己分组（含整组）；删别人的需管理员
    if not is_admin and target_owner != user["username"]:
        raise HTTPException(status_code=403, detail="只能删除自己的分组")

    db.audit(user["username"], _client_ip(request), "r2_delete",
             f"归属={target_owner} 分组={req.project_id} 批次={req.batch or '-'} 广告={req.ad_id or '-'}{' 仅分析' if req.analysis_only else ''}")

    pid = _epid(target_owner, req.project_id)      # R2 命名空间
    rawpid = pid
    bseg = f"{req.batch}/" if req.batch else ""   # 批次层
    if req.analysis_only:
        # 只删分析产物：analysis_output/ + 共性/横向（限定批次/广告）
        base = f"projects/{pid}/{bseg}"
        keys = [k for k in r2.list_keys(base)
                if "/analysis_output" in k or k.endswith("共性镜头prompt.json")
                or k.endswith("横向对比.md")]
        if req.ad_id:
            keys = [k for k in keys if k.startswith(f"{base}{req.ad_id}/")]
        deleted = r2.delete_keys(keys)
        scope = f"{base}{req.ad_id or ''} 的分析产物"
    elif req.ad_id:
        # 删单条广告：切片+分析产物 + 该广告的原视频 raw/（批次内）
        deleted = r2.delete_prefix(f"projects/{pid}/{bseg}{req.ad_id}/")
        deleted += r2.delete_keys(r2.list_keys(f"raw/{pid}/{bseg}{req.ad_id}.mp4"))
        scope = f"projects/{pid}/{bseg}{req.ad_id}/ (含原视频)"
    elif req.batch:
        # 删整批：该批次下 projects/ 与 raw/ 全清
        deleted = r2.delete_prefix(f"projects/{pid}/{req.batch}/")
        deleted += r2.delete_prefix(f"raw/{pid}/{req.batch}/")
        scope = f"批次 {req.batch}（projects/ + raw/）"
    else:
        # 删整个分组：projects/ 与 raw/ 全清，绝不留残留
        deleted = r2.delete_prefix(f"projects/{pid}/")
        deleted += r2.delete_prefix(f"raw/{pid}/")
        scope = f"projects/{pid}/ + raw/{pid}/"

    jobs_deleted = db.delete_jobs(req.project_id, req.batch or "", target_owner) if req.clear_jobs else 0
    return {"ok": True, "deleted": deleted, "scope": scope, "jobs_deleted": jobs_deleted}


class JobDeleteReq(BaseModel):
    job_id: int


@app.get("/download")
def download(project_id: str, ad_id: str = "", batch: str = "", owner: str = "", user: dict = Depends(current_user)):
    """打包某广告的全部产物为 zip：切片、shots.json、卖点(ad_aggregate)、
    脚本/镜头 prompt、分组级共性 prompt、以及首图(shot_000 第一帧)。"""
    import r2_storage as r2
    owner = (owner or "").strip() or user["username"]
    epid = _epid(owner, project_id)
    if ad_id:
        prefix = f"projects/{epid}/{batch}/{ad_id}/" if batch else f"projects/{epid}/{ad_id}/"
        dl_name = f"{project_id}_{batch+'_' if batch else ''}{ad_id}"
    else:                       # 整批打包
        prefix = f"projects/{epid}/{batch}/" if batch else f"projects/{epid}/"
        dl_name = f"{project_id}_{batch or 'all'}"
    keys = r2.list_keys(prefix)
    if not keys:
        raise HTTPException(status_code=404, detail="没有文件")

    import re as _re3
    work = Path(tempfile.mkdtemp(prefix="dl_"))
    tmp = work / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    label = batch or ""   # 批次 = 日期-产品名-序号

    def _arc(key, forced_ad):
        """给 zip 里的文件起可辨识的名字：批次_对象/ 批次_对象_镜头NNN.mp4"""
        rel = key[len(prefix):]
        parts = [p for p in rel.split("/") if p]
        if forced_ad:
            adname = forced_ad; inner = parts
        elif len(parts) >= 2:
            adname = parts[0]; inner = parts[1:]
        else:
            adname = ""; inner = parts
        folder = "_".join([p for p in [label, adname] if p]) or project_id
        fname = inner[-1] if inner else rel
        m = _re3.match(r"shot[_-]?(\d+)\.mp4$", fname, _re3.I)
        if m:
            return f"{folder}/{folder}_镜头{int(m.group(1)) + 1:03d}.mp4"
        if fname == "首图.jpg":
            return f"{folder}/{folder}_首图.jpg"
        return f"{folder}/" + "/".join(inner)

    zip_path = work / f"{dl_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for k in keys:
            if not k[len(prefix):]:
                continue
            dst = tmp / k.replace("/", "__")
            r2.download_file(k, dst)
            z.write(dst, _arc(k, ad_id or None))
        # 共性 prompt 若单条广告下载则补带上（整批下载时已在 keys 里）
        if ad_id:
            folder = "_".join([p for p in [label, ad_id] if p]) or project_id
            cbase = (f"projects/{epid}/{batch}/" if batch else f"projects/{epid}/")
            for cname in ("共性镜头prompt.json", "共性镜头prompt.en.json"):
                try:
                    dst = tmp / cname
                    r2.download_file(cbase + cname, dst)
                    z.write(dst, f"{folder}/{cname}")
                except Exception:
                    pass

    return FileResponse(
        zip_path, media_type="application/zip",
        filename=f"{dl_name}.zip",
        background=BackgroundTask(shutil.rmtree, str(work), ignore_errors=True),
    )


@app.post("/jobs/delete")
def jobs_delete(req: JobDeleteReq, request: Request, user: dict = Depends(current_user)):
    """删除单条任务记录（不动 R2 文件）。非管理员只能删自己的。"""
    n = db.delete_job(req.job_id, user["username"], user["role"] == "admin")
    if n:
        db.audit(user["username"], _client_ip(request), "job_delete", f"#{req.job_id}")
    return {"ok": True, "deleted": n}


@app.get("/download_file")
def download_file(url: str, name: str = "", user: dict = Depends(current_user)):
    """把 R2 公开文件以附件形式流式返回，强制浏览器下载（避免跨域直链变成在线播放）。"""
    base = getattr(r2, "PUBLIC_URL", "") or ""
    if not (url.startswith("https://") and (".r2.dev/" in url or (base and url.startswith(base)))):
        raise HTTPException(status_code=400, detail="非法下载地址")
    import requests
    from urllib.parse import quote
    try:
        rr = requests.get(url, stream=True, timeout=600)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"取文件失败: {e}")
    if rr.status_code != 200:
        raise HTTPException(status_code=502, detail=f"源返回 {rr.status_code}")
    fname = name or (url.split("/")[-1].split("?")[0]) or "download"
    return StreamingResponse(
        rr.iter_content(65536),
        media_type=rr.headers.get("content-type", "application/octet-stream"),
        headers={"Content-Disposition": "attachment; filename*=UTF-8''" + quote(fname)},
    )


# ── 收藏（团队共享可见；增删按 owner/admin 权限）──────────────────
class FavAddReq(BaseModel):
    kind: str                       # task / clip / prompt
    ref: str                        # 去重键
    label: Optional[str] = ""
    meta: Optional[dict] = None


class FavRemoveReq(BaseModel):
    id: int                         # 收藏行 id


@app.get("/fav/list")
def fav_list(user: dict = Depends(current_user)):
    """返回所有人的收藏（团队共享可见），带 owner；me 标注当前用户。"""
    return {"me": user["username"], "role": user["role"],
            "favorites": db.fav_list_all(), "db_enabled": db.enabled()}


@app.post("/fav/add")
def fav_add(req: FavAddReq, user: dict = Depends(current_user)):
    fid = db.fav_add(user["username"], req.kind, req.ref, req.label or "", req.meta or {})
    if fid is None:
        raise HTTPException(status_code=500, detail="收藏失败（未接数据库？）")
    return {"ok": True, "id": fid, "owner": user["username"], "ref": req.ref}


@app.post("/fav/remove")
def fav_remove(req: FavRemoveReq, user: dict = Depends(current_user)):
    return {"ok": True, "removed": db.fav_remove_by_id(req.id, user["username"], user["role"] == "admin")}


# ── 登录 / 账号 ──────────────────────────────────────────────────
class LoginReq(BaseModel):
    username: str
    pin: str = ""
    remember: bool = False


class ChangePinReq(BaseModel):
    old_pin: str = ""
    new_pin: str


@app.post("/auth/login")
def auth_login(req: LoginReq, request: Request):
    ip = _client_ip(request)
    u = db.user_verify(req.username.strip(), req.pin)
    if not u:
        db.audit(req.username.strip(), ip, "login_fail", "用户名或密码错误")
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = make_token(u["username"], u["role"], u["pin_ver"], req.remember)
    first = not db.audit_has_login(u["username"])
    db.audit(u["username"], ip, "login_first" if first else "login",
             ("首次登录 · " if first else "") + ("记住登录" if req.remember else "本次会话"))
    return {"ok": True, "token": token, "username": u["username"], "role": u["role"]}


@app.get("/auth/me")
def auth_me(request: Request, login: int = 0, user: dict = Depends(current_user)):
    # login=1：来自主页打开时的自动登录（记住 token），也算一次成功登录
    if login:
        first = not db.audit_has_login(user["username"])
        db.audit(user["username"], _client_ip(request), "login_first" if first else "login", "自动登录（打开网页）")
    return {"username": user["username"], "role": user["role"]}


@app.post("/auth/change_pin")
def auth_change_pin(req: ChangePinReq, user: dict = Depends(current_user)):
    ok, msg = db.user_change_pin(user["username"], req.old_pin, req.new_pin)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"ok": True}   # 前端应提示重新登录（旧 token 已失效）


# ── 用户管理（仅管理员）──────────────────────────────────────────
class UserCreateReq(BaseModel):
    username: str
    pin: str = ""
    role: str = "user"


class UserNameReq(BaseModel):
    username: str


class UserRoleReq(BaseModel):
    username: str
    role: str


class UserPinReq(BaseModel):
    username: str
    pin: str


class UserDisableReq(BaseModel):
    username: str
    disabled: bool = True


@app.get("/users")
def users_list(admin: dict = Depends(require_admin)):
    return {"users": db.user_list()}


@app.post("/users")
def users_create(req: UserCreateReq, request: Request, admin: dict = Depends(require_admin)):
    ok, msg = db.user_create(req.username, req.pin, req.role)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    db.audit(admin["username"], _client_ip(request), "user_create", f"{req.username}({req.role})")
    return {"ok": True}


@app.post("/users/role")
def users_role(req: UserRoleReq, request: Request, admin: dict = Depends(require_admin)):
    if req.username == admin["username"] and req.role != "admin":
        raise HTTPException(status_code=400, detail="不能取消自己的管理员权限")
    ok, msg = db.user_set_role(req.username, req.role)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    db.audit(admin["username"], _client_ip(request), "user_role", f"{req.username}→{req.role}")
    return {"ok": True}


@app.post("/users/reset_pin")
def users_reset_pin(req: UserPinReq, request: Request, admin: dict = Depends(require_admin)):
    ok, msg = db.user_reset_pin(req.username, req.pin)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    db.audit(admin["username"], _client_ip(request), "user_reset_pin", req.username)
    return {"ok": True}


@app.post("/users/disable")
def users_disable(req: UserDisableReq, request: Request, admin: dict = Depends(require_admin)):
    if req.username == admin["username"] and req.disabled:
        raise HTTPException(status_code=400, detail="不能禁用自己")
    ok, msg = db.user_set_disabled(req.username, req.disabled)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    db.audit(admin["username"], _client_ip(request), "user_disable", f"{req.username}={'禁用' if req.disabled else '启用'}")
    return {"ok": True}


@app.post("/users/delete")
def users_delete(req: UserNameReq, request: Request, admin: dict = Depends(require_admin)):
    if req.username == admin["username"]:
        raise HTTPException(status_code=400, detail="不能删除自己")
    if db.user_get(req.username) and db.user_get(req.username).get("role") == "admin" and db.admin_count() <= 1:
        raise HTTPException(status_code=400, detail="不能删除最后一个管理员")
    ok, msg = db.user_delete(req.username)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    db.audit(admin["username"], _client_ip(request), "user_delete", req.username)
    return {"ok": True}


# ── 管理员后台：审计日志 + 数据库查看 ────────────────────────────
@app.get("/admin/audit")
def admin_audit(limit: int = 200, username: str = "", action: str = "", admin: dict = Depends(require_admin)):
    return {"audit": db.audit_list(limit, username, action)}


@app.get("/admin/table")
def admin_table(name: str, limit: int = 200, owner: str = "", project: str = "", admin: dict = Depends(require_admin)):
    return db.table_rows(name, limit, owner, project)


class RowDelReq(BaseModel):
    name: str
    id: int


@app.post("/admin/table/delete")
def admin_table_delete(req: RowDelReq, request: Request, admin: dict = Depends(require_admin)):
    n = db.table_delete_row(req.name, req.id)
    if n:
        db.audit(admin["username"], _client_ip(request), "db_delete", f"{req.name}#{req.id}")
    return {"ok": True, "deleted": n}


@app.get("/admin/stats")
def admin_stats(admin: dict = Depends(require_admin)):
    return {"stats": db.db_stats(), "db_enabled": db.enabled()}


# ── 提示词修改（Gemini 重写 + 翻译，合为一个 prompt_edit 任务）─────────
class ReviseReq(BaseModel):
    project_id: str
    batch: str = ""
    ad: str = ""                       # shot / video 需要；common 不填
    ptype: str                         # shot / common / video
    idx: Optional[int] = None          # shot / common 的条目下标
    selling: Optional[List[str]] = None  # 直接修改的新卖点（shot/common）
    instruction: Optional[str] = ""    # 修改需求（间接）
    product: Optional[str] = ""
    owner: Optional[str] = None          # 归属用户；缺省=当前用户
    openrouter_key: Optional[str] = None
    deepseek_key: Optional[str] = None
    custom_prompts: Optional[dict] = None


def _find_key(prefix: str, suffix: str) -> Optional[str]:
    for k in r2.list_keys(prefix):
        if k.endswith(suffix) and not (suffix == "共性镜头prompt.json" and k.endswith(".en.json")):
            return k
    return None


def _run_revise_bg(req: "ReviseReq", job_id, owner=""):
    t0 = time.time()
    try:
        pid = _epid(owner, req.project_id)     # R2 命名空间
        batch, ad, ptype, idx = req.batch or "", req.ad or "", req.ptype, req.idx
        or_key = req.openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")
        ds_key = req.deepseek_key or DEEPSEEK_API_KEY

        # 1. 定位 zh json
        if ptype == "common":
            key_zh = _find_key(f"projects/{pid}/{batch}/", "共性镜头prompt.json")
            list_field = "共性镜头prompt列表"
        elif ptype == "shot":
            key_zh = _find_key(f"projects/{pid}/{batch}/{ad}/", "shot_prompts.json")
            list_field = "镜头级prompt列表"
        elif ptype == "video":
            key_zh = _find_key(f"projects/{pid}/{batch}/{ad}/", "video_prompt.json")
            list_field = None
        else:
            raise RuntimeError(f"未知 ptype: {ptype}")
        if not key_zh:
            raise RuntimeError("在 R2 找不到对应的提示词文件")
        data = r2.read_json(key_zh)

        # 2. 取当前条目内容
        if list_field:
            lst = data.get(list_field) or []
            if idx is None or idx < 0 or idx >= len(lst):
                raise RuntimeError(f"条目下标越界: {idx}")
            item = lst[idx]
            current = item.get("prompt", "")
            old_selling = item.get("卖点", []) or []
        else:
            current = data.get("video_prompt_text", "")
            old_selling = []

        # 3. Gemini 重写
        rev, or_cost, tin, tout = reviser.revise(
            current, req.product or data.get("产品名", ""), ptype,
            old_selling, req.selling, req.instruction or "", api_key=or_key)
        revised = rev.get("revised_prompt", "").strip()
        if not revised:
            raise RuntimeError("重写结果为空")
        # 最终卖点：直接改卖点优先；否则用 Gemini 返回的；再否则保持原样
        final_selling = (req.selling if req.selling is not None
                         else (rev.get("卖点") or old_selling))

        # 4. 写回 zh
        if list_field:
            if ptype in ("shot", "common"):
                item["卖点"] = final_selling
            item["prompt"] = revised
            lst[idx] = item
            data[list_field] = lst
        else:
            data["video_prompt_text"] = revised
        r2.upload_json(key_zh, data)

        # 5. 翻译并 patch .en.json
        ds_in = ds_out = 0
        if ds_key:
            try:
                instr = translator.pick_instr((req.custom_prompts or {}).get("translate"))
                ds_client = translator._client(ds_key)
                key_en = key_zh[:-len(".json")] + ".en.json"
                try:
                    en_data = r2.read_json(key_en)
                except Exception:
                    en_data = json.loads(json.dumps(data, ensure_ascii=False))  # 无则以 zh 为底
                if list_field:
                    sub = {k: item[k] for k in ("标题", "卖点", "prompt") if k in item}
                    en_sub, di, do = translator.translate_obj(sub, instr, ds_client)
                    ds_in += di; ds_out += do
                    en_lst = en_data.get(list_field) or []
                    while len(en_lst) <= idx:
                        en_lst.append({})
                    base = dict(en_lst[idx] or {})
                    base.update(en_sub or {})
                    en_lst[idx] = base
                    en_data[list_field] = en_lst
                else:
                    en_sub, di, do = translator.translate_obj(
                        {"video_prompt_text": revised}, instr, ds_client)
                    ds_in += di; ds_out += do
                    if en_sub:
                        en_data["video_prompt_text"] = en_sub.get("video_prompt_text", revised)
                r2.upload_json(key_en, en_data)
            except Exception as e:
                print(f"[revise] 翻译失败（不影响中文修改）: {e}")

        cost = or_cost + translator.cost_of(ds_in, ds_out)
        where = f"{ad or '共性'}" + (f" #{idx}" if idx is not None else "")
        db.finish_job(job_id, True, f"projects/{pid}/{batch}/",
                      f"已修改 {reviser.PTYPE_LABEL.get(ptype, ptype)}（{where}）并翻译",
                      time.time() - t0, cost_usd=cost,
                      tokens_in=tin + ds_in, tokens_out=tout + ds_out, cost_exact=False)
    except Exception as e:
        db.finish_job(job_id, False, "", f"{type(e).__name__}: {e}", time.time() - t0)


@app.post("/prompt/revise")
def prompt_revise(req: ReviseReq, user: dict = Depends(current_user)):
    if req.ptype not in ("shot", "common", "video"):
        raise HTTPException(status_code=400, detail="ptype 必须为 shot/common/video")
    ref = f"{req.ad or '共性'}" + (f"#{req.idx}" if req.idx is not None else "")
    target_owner = (req.owner or "").strip() or user["username"]
    if target_owner != user["username"] and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="只能修改自己分组的提示词")
    job_id = db.start_job("prompt_edit", req.project_id, ref, batch=req.batch or "", owner=user["username"])
    threading.Thread(target=_run_revise_bg, args=(req, job_id, target_owner), daemon=True).start()
    return {"ok": True, "started": True, "job_id": job_id}
