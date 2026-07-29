"""
素材生成 · 可插拔生成服务封装
================================
把"文/图 → 视频/图片"的第三方生成 API 收敛到一个统一接口，风格同 translator.py / reviser.py。
app.py 只依赖这里的 generate() / list_models() / estimate_usd() / providers_status()。

当前接入
--------
- **视频**：Seedance（seedanceapi.org v2）—— 一个 key 覆盖 seedance-2.0 / -fast / -mini 三个可选模型。
  这是目前**唯一可真正调用**的视频通道；fal 上的 Kling/Veo/Wan 作为占位（未配置，界面置灰）。
- **图片**：fal FLUX（占位，未配置；后续接 Seedream/FLUX 时填 key 即可）。

设计要点
--------
- **可插拔**：加服务只需加一个 provider 封装 + 目录项。
- **无 key 优雅降级**：没配对应 key 时不崩，调用抛 ProviderNotConfigured，界面显示"未配置"。
- **只用 requests**（已在 requirements 内），不新增依赖。
- provider 只负责"发起→轮询→拿到远端产物 URL"；下载并转存 R2 由 app.py 负责。

配置（Railway 变量）
--------------------
- SEEDANCE_KEY           Seedance API key（seedanceapi.org → Dashboard → API Keys）
- SEEDANCE_BASE          默认 https://seedanceapi.org/v2（一般不用改）
- FAL_KEY / REPLICATE_API_TOKEN   预留，接图片或其它视频模型时再填
"""
from __future__ import annotations
import math
import os
import time
from typing import Optional

import requests


# ── 异常 ─────────────────────────────────────────────────────────
class ProviderError(RuntimeError):
    pass


class ProviderNotConfigured(ProviderError):
    """未配置对应服务的 API key。"""
    pass


# ── 环境变量 ─────────────────────────────────────────────────────
SEEDANCE_KEY = os.environ.get("SEEDANCE_KEY", "") or os.environ.get("SEEDANCE_API_KEY", "")
SEEDANCE_BASE = os.environ.get("SEEDANCE_BASE", "https://seedanceapi.org/v2").rstrip("/")

FAL_KEY = os.environ.get("FAL_KEY", "") or os.environ.get("FAL_API_KEY", "")
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

GEN_IMAGE_PROVIDER = os.environ.get("GEN_IMAGE_PROVIDER", "fal")

_POLL_INTERVAL = float(os.environ.get("GEN_POLL_INTERVAL", "5"))
_POLL_TIMEOUT = float(os.environ.get("GEN_POLL_TIMEOUT", "600"))   # 10 分钟上限


# ── 模型目录 ─────────────────────────────────────────────────────
# 价格换算：Seedance 官方 1 USD = 100 credits。下表 usd = credits / 100。
# video 计价：price_kind="table" 按时长查表；"mini" 按 每秒费率×秒数（向上取整积分）。
# aspect/resolution/durations 供前端渲染选择器。
MODEL_CATALOG = {
    # ── 视频（Seedance，可调用）──
    "seedance-2.0": {
        "kind": "video", "provider": "seedance", "seedance_model": "seedance-2.0",
        "label": "Seedance 2.0",
        "family": "Seedance 2.0", "family_desc": "ByteDance 多模态视频生成",
        "version": "Normal", "version_desc": "标准 · 高质量",
        "durations": [5, 10, 15], "resolutions": ["480p", "720p", "1080p"],
        "aspects": ["16:9", "9:16", "4:3", "3:4"], "audio": False,
        "price_kind": "table", "table": {5: 2.40, 10: 4.80, 15: 7.20},
    },
    "seedance-2.0-fast": {
        "kind": "video", "provider": "seedance", "seedance_model": "seedance-2.0-fast",
        "label": "Seedance 2.0 Fast",
        "family": "Seedance 2.0", "family_desc": "ByteDance 多模态视频生成",
        "version": "Fast", "version_desc": "更快 · 更省",
        "durations": [5, 10, 15], "resolutions": ["480p", "720p", "1080p"],
        "aspects": ["16:9", "9:16", "4:3", "3:4"], "audio": False,
        "price_kind": "table", "table": {5: 1.60, 10: 3.20, 15: 4.80},
    },
    "seedance-2-mini": {
        "kind": "video", "provider": "seedance", "seedance_model": "seedance-2-mini",
        "label": "Seedance Mini",
        "family": "Seedance Mini", "family_desc": "灵活时长 · 最省",
        "version": "Mini", "version_desc": "4–15 秒 · 480/720p",
        "durations": [4, 5, 8, 10, 15], "resolutions": ["480p", "720p"],
        "aspects": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], "audio": False,
        "price_kind": "mini",
        "mini_rate": {"480p": 0.1425, "720p": 0.3075},  # USD/秒（credits/100/秒）
    },
    # ── 图片（占位，未配置）──
    "flux-pro": {
        "kind": "image", "provider": "fal", "label": "FLUX Pro",
        "family": "FLUX Pro", "family_desc": "高质量图片生成（待配置）",
        "version": "Standard", "version_desc": "标准",
        "endpoint": "fal-ai/flux-pro/v1.1",
        "usd_per_image": 0.05,
        "resolutions": ["square_hd", "portrait_16_9", "landscape_16_9", "portrait_4_3", "landscape_4_3"],
        "aspects": [], "audio": False,
    },
}


def _provider_configured(provider: str) -> bool:
    if provider == "seedance":
        return bool(SEEDANCE_KEY)
    if provider == "fal":
        return bool(FAL_KEY)
    if provider == "replicate":
        return bool(REPLICATE_API_TOKEN)
    return False


def list_models(kind: Optional[str] = None) -> list:
    """返回模型目录（供前端渲染选择器）。附带该 provider 是否已配置。"""
    out = []
    for mid, m in MODEL_CATALOG.items():
        if kind and m["kind"] != kind:
            continue
        out.append({
            "id": mid,
            "kind": m["kind"],
            "label": m["label"],
            "family": m.get("family", m["label"]),
            "family_desc": m.get("family_desc", ""),
            "version": m.get("version", ""),
            "version_desc": m.get("version_desc", ""),
            "provider": m["provider"],
            "configured": _provider_configured(m["provider"]),
            "durations": m.get("durations", []),
            "resolutions": m.get("resolutions", []),
            "aspects": m.get("aspects", []),
            "audio": m.get("audio", False),
            # 定价信息（前端据此实时估算）
            "price_kind": m.get("price_kind"),
            "table": {str(k): v for k, v in (m.get("table") or {}).items()},
            "mini_rate": m.get("mini_rate"),
            "usd_per_image": m.get("usd_per_image"),
        })
    return out


def providers_status() -> dict:
    return {
        "seedance": bool(SEEDANCE_KEY),
        "fal": bool(FAL_KEY),
        "replicate": bool(REPLICATE_API_TOKEN),
        "image_provider": GEN_IMAGE_PROVIDER,
    }


# ── 成本估算（USD）─────────────────────────────────────────────────
def estimate_usd(model: str, params: dict) -> float:
    m = MODEL_CATALOG.get(model)
    if not m:
        return 0.0
    if m["kind"] == "video":
        dur = int(params.get("duration") or (m.get("durations") or [5])[0])
        pk = m.get("price_kind")
        if pk == "table":
            tbl = m.get("table") or {}
            if dur in tbl:
                return round(float(tbl[dur]), 4)
            # 不在表内则按最接近时长的每秒价折算
            per = min(tbl.items(), key=lambda kv: abs(kv[0] - dur))
            return round(per[1] / per[0] * dur, 4)
        if pk == "mini":
            res = str(params.get("resolution") or "720p")
            rate = (m.get("mini_rate") or {}).get(res, (m.get("mini_rate") or {}).get("720p", 0))
            # 官方：向上取整到积分（credits=usd*100）
            credits = math.ceil(rate * dur * 100)
            return round(credits / 100.0, 4)
        return 0.0
    else:
        n = int(params.get("num", 1) or 1)
        return round(float(m.get("usd_per_image", 0)) * n, 4)


# ── 统一入口 ─────────────────────────────────────────────────────
def generate(model: str, prompt: str, params: dict, ref_image_urls=None) -> list:
    """发起一次生成，阻塞直到完成，返回 [{"url":..,"kind":"video|image"}]。
    ref_image_urls：参考图 URL 列表（首帧 + 其它参考图，最多 4 张）。
    失败抛 ProviderError；未配置抛 ProviderNotConfigured。"""
    refs = [u for u in (ref_image_urls or []) if u]
    m = MODEL_CATALOG.get(model)
    if not m:
        raise ProviderError(f"未知模型：{model}")
    prov = m["provider"]
    if not _provider_configured(prov):
        raise ProviderNotConfigured(f"服务 {prov} 未配置 API key，请在 Railway 变量里填写后重试")
    if prov == "seedance":
        return _seedance_generate(m, prompt, params, refs)
    if prov == "fal":
        return _fal_generate(m, prompt, params, refs)
    if prov == "replicate":
        return _replicate_generate(m, prompt, params, refs)
    raise ProviderError(f"不支持的 provider：{prov}")


# ── Seedance 实现（seedanceapi.org v2）──────────────────────────
# POST {base}/generate  → {code,data:{task_id,status,consumed_credits}}
# GET  {base}/status?task_id=  → {code,data:{status:SUCCESS/IN_PROGRESS/FAILED,response:[url],error_message}}
def _seedance_headers() -> dict:
    return {"Authorization": f"Bearer {SEEDANCE_KEY}", "Content-Type": "application/json"}


def _seedance_generate(m: dict, prompt: str, params: dict, ref_image_urls=None) -> list:
    # 严格按 seedanceapi.org v2 手册：prompt/model/duration 必填；resolution 可选；
    # 文生视频用 aspect_ratio；图生视频用 images[]（1–4 张，首张=首帧）。
    refs = [u for u in (ref_image_urls or []) if u][:4]
    dur = int(params.get("duration") or (m.get("durations") or [5])[0])
    if dur not in (m.get("durations") or [dur]):
        dur = (m.get("durations") or [5])[0]        # 兜底到合法时长
    body = {"prompt": prompt, "model": m["seedance_model"], "duration": dur}
    if params.get("resolution"):
        body["resolution"] = params["resolution"]
    if refs:
        body["images"] = refs                        # 图生视频（含多参考图）
    elif params.get("aspect"):
        body["aspect_ratio"] = params["aspect"]      # 文生视频才需画幅

    try:
        r = requests.post(f"{SEEDANCE_BASE}/generate", headers=_seedance_headers(),
                          json=body, timeout=60)
    except Exception as e:
        raise ProviderError(f"Seedance 请求失败：{e}")
    if r.status_code == 401:
        raise ProviderError("Seedance API key 无效（401）")
    if r.status_code == 402:
        raise ProviderError("Seedance 账户积分不足（402），请到 seedanceapi.org 充值")
    if r.status_code >= 400:
        raise ProviderError(f"Seedance 返回 {r.status_code}：{r.text[:300]}")
    data = (r.json() or {}).get("data") or {}
    task_id = data.get("task_id")
    if not task_id:
        raise ProviderError(f"Seedance 未返回 task_id：{r.text[:300]}")

    # 轮询
    t0 = time.time()
    while True:
        if time.time() - t0 > _POLL_TIMEOUT:
            raise ProviderError("Seedance 生成超时")
        time.sleep(_POLL_INTERVAL)
        try:
            s = requests.get(f"{SEEDANCE_BASE}/status", headers=_seedance_headers(),
                             params={"task_id": task_id}, timeout=30).json()
        except Exception as e:
            raise ProviderError(f"Seedance 轮询失败：{e}")
        sd = (s or {}).get("data") or {}
        st = (sd.get("status") or "").upper()
        if st == "SUCCESS":
            urls = sd.get("response") or []
            if isinstance(urls, str):
                urls = [urls]
            outs = [{"url": u, "kind": "video"} for u in urls if u]
            if not outs:
                raise ProviderError("Seedance 成功但无产物 URL")
            return outs
        if st in ("FAILED", "ERROR"):
            raise ProviderError(f"Seedance 生成失败：{sd.get('error_message') or st}")
        # IN_PROGRESS → 继续轮询


# ── fal.ai 实现（预留，图片/其它视频模型接入时启用）────────────────
_FAL_BASE = "https://queue.fal.run"


def _fal_headers() -> dict:
    return {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}


def _fal_generate(m: dict, prompt: str, params: dict, ref_image_urls=None) -> list:
    ref = ([u for u in (ref_image_urls or []) if u] or [""])[0]
    endpoint = m["endpoint"]
    body = {"prompt": prompt}
    if m["kind"] == "image":
        if params.get("resolution"):
            body["image_size"] = params["resolution"]
        n = int(params.get("num", 1) or 1)
        if n > 1:
            body["num_images"] = n
        if ref:
            body["image_url"] = ref
    try:
        r = requests.post(f"{_FAL_BASE}/{endpoint}", headers=_fal_headers(), json=body, timeout=60)
    except Exception as e:
        raise ProviderError(f"fal 请求失败：{e}")
    if r.status_code >= 400:
        raise ProviderError(f"fal 返回 {r.status_code}：{r.text[:300]}")
    data = r.json()
    status_url = data.get("status_url")
    response_url = data.get("response_url")
    if not status_url or not response_url:
        return _fal_parse_outputs(data)
    t0 = time.time()
    while True:
        if time.time() - t0 > _POLL_TIMEOUT:
            raise ProviderError("fal 生成超时")
        time.sleep(_POLL_INTERVAL)
        s = requests.get(status_url, headers=_fal_headers(), timeout=30).json()
        st = (s.get("status") or "").upper()
        if st in ("COMPLETED", "OK", "SUCCESS"):
            break
        if st in ("FAILED", "ERROR", "CANCELLED"):
            raise ProviderError(f"fal 生成失败：{str(s)[:300]}")
    result = requests.get(response_url, headers=_fal_headers(), timeout=60).json()
    return _fal_parse_outputs(result)


def _fal_parse_outputs(result: dict) -> list:
    out = []
    if not isinstance(result, dict):
        return out
    for it in (result.get("images") or []):
        u = it.get("url") if isinstance(it, dict) else it
        if u:
            out.append({"url": u, "kind": "image"})
    v = result.get("video")
    if isinstance(v, dict) and v.get("url"):
        out.append({"url": v["url"], "kind": "video"})
    if not out:
        raise ProviderError(f"fal 结果无法解析产物：{str(result)[:300]}")
    return out


# ── Replicate 实现（预留）───────────────────────────────────────
def _replicate_generate(m: dict, prompt: str, params: dict, ref_image_urls=None) -> list:
    ref = ([u for u in (ref_image_urls or []) if u] or [""])[0]
    headers = {"Authorization": f"Bearer {REPLICATE_API_TOKEN}",
               "Content-Type": "application/json", "Prefer": "wait"}
    inp = {"prompt": prompt}
    if ref:
        inp["image"] = ref
    url = f"https://api.replicate.com/v1/models/{m['replicate_model']}/predictions"
    r = requests.post(url, headers=headers, json={"input": inp}, timeout=120)
    if r.status_code >= 400:
        raise ProviderError(f"replicate 返回 {r.status_code}：{r.text[:300]}")
    data = r.json()
    t0 = time.time()
    while (data.get("status") or "") not in ("succeeded", "failed", "canceled"):
        if time.time() - t0 > _POLL_TIMEOUT:
            raise ProviderError("replicate 生成超时")
        get_url = (data.get("urls") or {}).get("get")
        if not get_url:
            break
        time.sleep(_POLL_INTERVAL)
        data = requests.get(get_url, headers=headers, timeout=30).json()
    if (data.get("status") or "") == "failed":
        raise ProviderError(f"replicate 生成失败：{str(data.get('error'))[:300]}")
    out, output = [], data.get("output")
    for u in (output if isinstance(output, list) else [output]):
        if isinstance(u, str):
            out.append({"url": u, "kind": m["kind"]})
    if not out:
        raise ProviderError(f"replicate 结果无法解析产物：{str(output)[:300]}")
    return out
