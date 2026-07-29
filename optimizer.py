"""
素材生成 · 提示词 AI 优化（OpenRouter / Gemini）
================================================
把用户手写的、或从广告分析结果带过来的提示词，改写为一条**符合广告视频/图片素材生成标准**、
可直接投喂生成模型的高质量提示词。每次优化都独立记录（优化要求 + 优化后文本），供任务留档。

与 reviser.py 类似，走 OpenRouter（与分析同一把 key）。只负责"优化这一条"，保存与编排由 app.py 负责。
"""
from __future__ import annotations
import os
import json
import re
from typing import Optional

OPTIMIZE_MODEL = os.environ.get("OPTIMIZE_MODEL", "google/gemini-2.5-flash")

_ELEMENTS = {
    "video": "景别与运镜、场景环境、主体位置与动作过程、产品呈现与卖点表现、光位/材质/速度、时长与节奏、画面文字（如有则忠实保留）",
    "image": "构图与视角、场景环境、主体与产品摆位、材质与光影、风格与色调、画面文字（如有则忠实保留）",
}

OPTIMIZE_PROMPT = """你是资深广告 AI 素材生成 prompt 工程师，服务电商/短视频广告投放。
请把下面的【原始提示词】改写为一条**可直接投喂 AI {media}生成模型**的高质量提示词。

【原始提示词】：
{text}

【额外优化要求】（若非空，必须满足）：{instruction}

【改写标准】（面向广告{media}素材）：
- 用**镜头语言/画面语言**客观描述画面，不写解说旁白、不写分镜小标题。
- 要素完整、结构清晰，覆盖：{elements}。
- 让主体、产品、卖点被"演"出来：用动作、特写、光影把卖点直观呈现，而非空泛形容词堆砌。
- 画面可执行、可复刻，保留必要的技术参数；语言简洁有力。
- 保持原意与卖点不跑偏，只做增强与规范化。中文输出。

只输出如下 JSON，不要 markdown、不要多余文字：
{{"optimized_prompt": "优化后的完整提示词"}}"""


def _client(api_key: Optional[str] = None):
    from openai import OpenAI
    import httpx
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("缺少 OPENROUTER_API_KEY")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                  http_client=httpx.Client(timeout=300.0),
                  default_headers={"HTTP-Referer": "https://local.prompt-optimizer",
                                   "X-Title": "Prompt Optimizer"})


def _usage_cost(usage) -> float:
    if usage is None:
        return 0.0
    c = getattr(usage, "cost", None)
    if c is None:
        c = (getattr(usage, "model_extra", None) or {}).get("cost")
    try:
        return float(c or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _extract_json(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw).rstrip("`").strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def optimize(text: str, instruction: str = "", kind: str = "video",
             api_key: Optional[str] = None, model: str = None):
    """优化一条提示词。返回 (optimized_text, cost, tin, tout)。失败抛异常。"""
    if not (text or "").strip():
        raise RuntimeError("待优化的提示词为空")
    media = "图片" if kind == "image" else "视频"
    model = model or OPTIMIZE_MODEL
    client = _client(api_key)
    prompt = OPTIMIZE_PROMPT.format(
        media=media,
        text=text.strip(),
        instruction=(instruction or "").strip() or "（无，按标准整体优化）",
        elements=_ELEMENTS.get(kind, _ELEMENTS["video"]),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    txt = resp.choices[0].message.content or ""
    data = _extract_json(txt)
    opt = (data.get("optimized_prompt") or "").strip() if isinstance(data, dict) else ""
    if not opt:
        raise RuntimeError("优化返回缺少 optimized_prompt")
    u = getattr(resp, "usage", None)
    tin = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
    tout = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
    return opt, _usage_cost(u), tin, tout
