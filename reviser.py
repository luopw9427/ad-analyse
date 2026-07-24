"""
提示词修改（Gemini / OpenRouter）
==================================

对某一条已生成的提示词（视频级 / 单镜头 / 共性镜头）做二次修改：
- 直接改卖点：让提示词紧扣新卖点重写；
- 修改需求：按自由文本需求调整（可能顺带调整卖点，也可能不动）。

只负责"重写这一条"，写回 R2 + 翻译由 app.py 串起来。客户端走 OpenRouter（与分析同一把 key）。
"""

from __future__ import annotations
import os
import json
import re
from typing import Optional

REVISE_MODEL = os.environ.get("REVISE_MODEL", "google/gemini-2.5-flash")

# ptype → 中文说明
PTYPE_LABEL = {"video": "视频级脚本 prompt", "shot": "单镜头 prompt", "common": "共性镜头 prompt"}

REVISE_PROMPT = """你是资深广告 AI 视频生成 prompt 工程师。下面给你一条【当前提示词】（{ptype_label}），
请按要求修改它，修改后仍是可直接投喂 AI 视频生成模型的高质量镜头语言 prompt。

产品：{product}

【当前提示词】：
{current}

【当前卖点】：{old_selling}
【目标卖点】（若非空，修改后的提示词必须紧扣这些卖点来呈现画面）：{new_selling}
【额外修改需求】（若非空，必须满足）：{instruction}

【硬性要求】
- 保持原有结构与镜头语言风格：只写镜头语言、不写解说；不复述参考图内容（用"与图片X一致"）；
  保留景别 / 运镜 / 技术参数 / 主体位置与动作 / 产品呈现 / 光位 / 速度 / 字幕等可复刻要素；
  镜头不加小标题。
- 若给了目标卖点：让镜头的主体动作、产品呈现、景别与特写直接把这些卖点"演"出来。
- 若给了修改需求：在不破坏整体质量的前提下满足它。
- 只改必要之处，其余尽量保留，时长/时段等结构不要无故变动。
- 同时给出修改后的提示词最终体现的卖点列表（可在目标卖点基础上增删；视频级可留空数组）。

只输出如下 JSON，不要 markdown、不要多余文字：
{{"revised_prompt": "修改后的完整提示词文本", "卖点": ["卖点1", "卖点2"]}}"""


def _client(api_key: Optional[str] = None):
    from openai import OpenAI
    import httpx
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("缺少 OPENROUTER_API_KEY")
    http_client = httpx.Client(timeout=300.0)
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                 http_client=http_client,
                 default_headers={"HTTP-Referer": "https://local.prompt-reviser",
                                  "X-Title": "Prompt Reviser"})


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


def revise(current: str, product: str, ptype: str,
           old_selling, new_selling, instruction: str,
           api_key: Optional[str] = None, model: str = None):
    """重写一条提示词。返回 (dict{revised_prompt, 卖点}, cost, tin, tout)。失败抛异常。"""
    model = model or REVISE_MODEL
    client = _client(api_key)
    prompt = REVISE_PROMPT.format(
        ptype_label=PTYPE_LABEL.get(ptype, "prompt"),
        product=product or "未知产品",
        current=current or "",
        old_selling="、".join(old_selling or []) or "（无）",
        new_selling="、".join(new_selling or []) or "（未指定，保持或按需求调整）",
        instruction=(instruction or "").strip() or "（无）",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    txt = resp.choices[0].message.content or ""
    data = _extract_json(txt)
    if not isinstance(data, dict) or "revised_prompt" not in data:
        raise RuntimeError("Gemini 返回缺少 revised_prompt")
    cost = _usage_cost(getattr(resp, "usage", None))
    u = getattr(resp, "usage", None)
    tin = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
    tout = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
    data.setdefault("卖点", [])
    return data, cost, tin, tout
