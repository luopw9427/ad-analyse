"""
素材生成 · 可插拔生成服务封装
================================
把"文/图 → 视频/图片"的第三方生成 API 收敛到一个统一接口，风格同 translator.py / reviser.py：
一个模块封装一个/多个服务客户端，app.py 只依赖这里的 generate() / list_models() / estimate_usd()。

设计要点
--------
- **可插拔**：视频与图片可用不同 provider（默认视频=fal、图片=fal，可切 replicate）。
- **无 key 优雅降级**：没配 FAL_KEY / REPLICATE_API_TOKEN 时不崩，调用时抛 ProviderNotConfigured，
  前端/后台照常展示"未配置"，不影响提示词抽取模块。
- **只用 requests**（已在 requirements 内），不新增依赖。
- provider 只负责"发起→轮询→拿到远端产物 URL"；下载并转存 R2 由 app.py 负责（复用现有 R2 封装）。

模型目录（MODEL_CATALOG）里每个模型带：kind / provider / 计价 / 参数可选项，
前端据此渲染选择器，后端据此估算成本（成本→积分换算在 app.py）。

获取 API key 后：在 Railway 变量里填 FAL_KEY（和/或 REPLICATE_API_TOKEN）即可，无需改代码。
"""
from __future__ import annotations
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
FAL_KEY = os.environ.get("FAL_KEY", "") or os.environ.get("FAL_API_KEY", "")
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

# 默认 provider（可用 Railway 变量覆盖）
GEN_VIDEO_PROVIDER = os.environ.get("GEN_VIDEO_PROVIDER", "fal")   # fal
GEN_IMAGE_PROVIDER = os.environ.get("GEN_IMAGE_PROVIDER", "fal")   # fal / replicate

# 轮询设置
_POLL_INTERVAL = float(os.environ.get("GEN_POLL_INTERVAL", "3"))
_POLL_TIMEOUT = float(os.environ.get("GEN_POLL_TIMEOUT", "600"))   # 10 分钟上限


# ── 模型目录 ─────────────────────────────────────────────────────
# 价格为「参考单价（USD）」，2026 年市场价，会浮动；用于成本估算，实际以 API 返回/账单为准。
# video: usd_per_sec 按秒计；image: usd_per_image 按张计。
# endpoint = fal 模型路径；replicate_model = "owner/name"。
MODEL_CATALOG = {
    # ── 视频 ──
    "wan-2.5": {
        "kind": "video", "provider": "fal", "label": "Wan 2.5（最省，跑通首选）",
        "endpoint": "fal-ai/wan-25-preview/text-to-video",
        "endpoint_i2v": "fal-ai/wan-25-preview/image-to-video",
        "usd_per_sec": 0.05,
        "durations": [5, 10], "resolutions": ["480p", "720p", "1080p"],
        "aspects": ["16:9", "9:16", "1:1"], "audio": True,
    },
    "kling-v2": {
        "kind": "video", "provider": "fal", "label": "Kling v2（性价比标杆·推荐）",
        "endpoint": "fal-ai/kling-video/v2/standard/text-to-video",
        "endpoint_i2v": "fal-ai/kling-video/v2/standard/image-to-video",
        "usd_per_sec": 0.07,
        "durations": [5, 10], "resolutions": ["720p", "1080p"],
        "aspects": ["16:9", "9:16", "1:1"], "audio": False,
    },
    "veo-3.1": {
        "kind": "video", "provider": "fal", "label": "Veo 3.1（最强·带原生音频/4K·最贵）",
        "endpoint": "fal-ai/veo3.1/text-to-video",
        "endpoint_i2v": "fal-ai/veo3.1/image-to-video",
        "usd_per_sec": 0.75,
        "durations": [5, 8], "resolutions": ["720p", "1080p", "4k"],
        "aspects": ["16:9", "9:16"], "audio": True,
    },
    # ── 图片 ──
    "flux-pro": {
        "kind": "image", "provider": "fal", "label": "FLUX Pro（高质量）",
        "endpoint": "fal-ai/flux-pro/v1.1",
        "usd_per_image": 0.05,
        "resolutions": ["square_hd", "portrait_16_9", "landscape_16_9", "portrait_4_3", "landscape_4_3"],
        "aspects": [], "audio": False,
    },
    "flux-schnell": {
        "kind": "image", "provider": "fal", "label": "FLUX Schnell（快·便宜）",
        "endpoint": "fal-ai/flux/schnell",
        "usd_per_image": 0.003,
        "resolutions": ["square_hd", "portrait_16_9", "landscape_16_9", "portrait_4_3", "landscape_4_3"],
        "aspects": [], "audio": False,
    },
    "flux-replicate": {
        "kind": "image", "provider": "replicate", "label": "FLUX（Replicate 备用）",
        "replicate_model": "black-forest-labs/flux-1.1-pro",
        "usd_per_image": 0.04,
        "resolutions": ["1:1", "16:9", "9:16", "4:3", "3:4"],
        "aspects": [], "audio": False,
    },
}


def list_models(kind: Optional[str] = None) -> list:
    """返回模型目录（供前端渲染选择器）。kind 可过滤 video/image。
    附带该 provider 是否已配置 key，未配置的前端可置灰。"""
    out = []
    for mid, m in MODEL_CATALOG.items():
        if kind and m["kind"] != kind:
            continue
        prov = m["provider"]
        out.append({
            "id": mid,
            "kind": m["kind"],
            "label": m["label"],
            "provider": prov,
            "configured": _provider_configured(prov),
            "usd_per_sec": m.get("usd_per_sec"),
            "usd_per_image": m.get("usd_per_image"),
            "durations": m.get("durations", []),
            "resolutions": m.get("resolutions", []),
            "aspects": m.get("aspects", []),
            "audio": m.get("audio", False),
        })
    return out


def _provider_configured(provider: str) -> bool:
    if provider == "fal":
        return bool(FAL_KEY)
    if provider == "replicate":
        return bool(REPLICATE_API_TOKEN)
    return False


def providers_status() -> dict:
    return {
        "fal": bool(FAL_KEY),
        "replicate": bool(REPLICATE_API_TOKEN),
        "video_provider": GEN_VIDEO_PROVIDER,
        "image_provider": GEN_IMAGE_PROVIDER,
    }


# ── 成本估算 ─────────────────────────────────────────────────────
def estimate_usd(model: str, params: dict) -> float:
    """按模型×参数估算一次生成的美元成本（用于积分预扣；实际以账单为准）。"""
    m = MODEL_CATALOG.get(model)
    if not m:
        return 0.0
    n = int(params.get("num", 1) or 1)
    if m["kind"] == "video":
        dur = float(params.get("duration", (m.get("durations") or [5])[0]))
        base = float(m.get("usd_per_sec", 0)) * dur
        # 分辨率粗略加成
        res = str(params.get("resolution", "720p"))
        mult = {"480p": 0.7, "720p": 1.0, "1080p": 1.6, "4k": 3.0}.get(res, 1.0)
        return round(base * mult * n, 4)
    else:
        return round(float(m.get("usd_per_image", 0)) * n, 4)


# ── 统一入口 ─────────────────────────────────────────────────────
def generate(model: str, prompt: str, params: dict, ref_image_url: str = "") -> list:
    """发起一次生成，阻塞直到完成，返回产物远端 URL 列表 [{"url":..,"kind":"video|image"}]。
    失败抛 ProviderError；未配置抛 ProviderNotConfigured。app.py 负责把这些 URL 下载转存 R2。"""
    m = MODEL_CATALOG.get(model)
    if not m:
        raise ProviderError(f"未知模型：{model}")
    prov = m["provider"]
    if not _provider_configured(prov):
        raise ProviderNotConfigured(f"服务 {prov} 未配置 API key，请在 Railway 变量里填写后重试")
    if prov == "fal":
        return _fal_generate(m, prompt, params, ref_image_url)
    if prov == "replicate":
        return _replicate_generate(m, prompt, params, ref_image_url)
    raise ProviderError(f"不支持的 provider：{prov}")


# ── fal.ai 实现（queue REST API）────────────────────────────────
# 文档：POST https://queue.fal.run/{endpoint} → {request_id,status_url,response_url}
#       轮询 status_url 直到 status=COMPLETED → GET response_url 取结果。
_FAL_BASE = "https://queue.fal.run"


def _fal_headers() -> dict:
    return {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}


def _fal_generate(m: dict, prompt: str, params: dict, ref_image_url: str = "") -> list:
    endpoint = m["endpoint"]
    body = {"prompt": prompt}
    if m["kind"] == "video":
        if params.get("duration"):
            body["duration"] = str(int(params["duration"]))
        if params.get("resolution"):
            body["resolution"] = params["resolution"]
        if params.get("aspect"):
            body["aspect_ratio"] = params["aspect"]
        if m.get("audio") and params.get("audio") is not None:
            body["generate_audio"] = bool(params["audio"])
        if ref_image_url and m.get("endpoint_i2v"):
            endpoint = m["endpoint_i2v"]
            body["image_url"] = ref_image_url
    else:  # image
        if params.get("resolution"):
            body["image_size"] = params["resolution"]
        n = int(params.get("num", 1) or 1)
        if n > 1:
            body["num_images"] = n
        if ref_image_url:
            body["image_url"] = ref_image_url  # 图生图/垫图（部分端点支持）

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
        # 有些同步端点直接返回结果
        return _fal_parse_outputs(data)

    # 轮询
    t0 = time.time()
    while True:
        if time.time() - t0 > _POLL_TIMEOUT:
            raise ProviderError("fal 生成超时")
        time.sleep(_POLL_INTERVAL)
        try:
            s = requests.get(status_url, headers=_fal_headers(), timeout=30).json()
        except Exception as e:
            raise ProviderError(f"fal 轮询失败：{e}")
        st = (s.get("status") or "").upper()
        if st in ("COMPLETED", "OK", "SUCCESS"):
            break
        if st in ("FAILED", "ERROR", "CANCELLED"):
            raise ProviderError(f"fal 生成失败：{str(s)[:300]}")
    try:
        result = requests.get(response_url, headers=_fal_headers(), timeout=60).json()
    except Exception as e:
        raise ProviderError(f"fal 取结果失败：{e}")
    return _fal_parse_outputs(result)


def _fal_parse_outputs(result: dict) -> list:
    """从 fal 结果里尽量鲁棒地抽取产物 URL。"""
    out = []
    if not isinstance(result, dict):
        return out
    # 视频常见：{"video":{"url":..}} 或 {"video":{...}, ...}
    v = result.get("video")
    if isinstance(v, dict) and v.get("url"):
        out.append({"url": v["url"], "kind": "video"})
    elif isinstance(v, str):
        out.append({"url": v, "kind": "video"})
    for key in ("videos",):
        for it in (result.get(key) or []):
            u = it.get("url") if isinstance(it, dict) else it
            if u:
                out.append({"url": u, "kind": "video"})
    # 图片常见：{"images":[{"url":..}]}
    for it in (result.get("images") or []):
        u = it.get("url") if isinstance(it, dict) else it
        if u:
            out.append({"url": u, "kind": "image"})
    img = result.get("image")
    if isinstance(img, dict) and img.get("url"):
        out.append({"url": img["url"], "kind": "image"})
    if not out:
        raise ProviderError(f"fal 结果无法解析产物：{str(result)[:300]}")
    return out


# ── Replicate 实现 ──────────────────────────────────────────────
# 文档：POST https://api.replicate.com/v1/models/{owner}/{name}/predictions
#       Header Prefer: wait 可等待完成；否则轮询 get URL。
def _replicate_generate(m: dict, prompt: str, params: dict, ref_image_url: str = "") -> list:
    model_path = m["replicate_model"]
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }
    inp = {"prompt": prompt}
    if params.get("resolution"):
        inp["aspect_ratio"] = params["resolution"]
    n = int(params.get("num", 1) or 1)
    if n > 1:
        inp["num_outputs"] = n
    if ref_image_url:
        inp["image"] = ref_image_url
    url = f"https://api.replicate.com/v1/models/{model_path}/predictions"
    try:
        r = requests.post(url, headers=headers, json={"input": inp}, timeout=120)
    except Exception as e:
        raise ProviderError(f"replicate 请求失败：{e}")
    if r.status_code >= 400:
        raise ProviderError(f"replicate 返回 {r.status_code}：{r.text[:300]}")
    data = r.json()

    # Prefer: wait 可能已完成；否则轮询
    t0 = time.time()
    while (data.get("status") or "") not in ("succeeded", "failed", "canceled"):
        if time.time() - t0 > _POLL_TIMEOUT:
            raise ProviderError("replicate 生成超时")
        get_url = (data.get("urls") or {}).get("get")
        if not get_url:
            break
        time.sleep(_POLL_INTERVAL)
        try:
            data = requests.get(get_url, headers=headers, timeout=30).json()
        except Exception as e:
            raise ProviderError(f"replicate 轮询失败：{e}")
    if (data.get("status") or "") == "failed":
        raise ProviderError(f"replicate 生成失败：{str(data.get('error'))[:300]}")

    kind = m["kind"]
    out = []
    output = data.get("output")
    if isinstance(output, str):
        out.append({"url": output, "kind": kind})
    elif isinstance(output, list):
        for u in output:
            if isinstance(u, str):
                out.append({"url": u, "kind": kind})
    if not out:
        raise ProviderError(f"replicate 结果无法解析产物：{str(output)[:300]}")
    return out
