"""
提示词翻译（DeepSeek）
=======================

把「合并流程」产出的中文结构化提示词/卖点，翻译成英文，供英文团队直接投喂
AI 视频生成模型。核心约束：**只翻译值里的自然语言，键名/结构/数值一律保留**，
这样英文 JSON 与中文 JSON 字段完全一致，前端可复用同一套渲染。

- 客户端：OpenAI 兼容，base_url = https://api.deepseek.com（与 OpenRouter 管理方式一致）
- 并发：由调用方用线程池驱动，本模块只负责「翻一个对象」
- 提示词 TRANSLATE_PROMPT：作为可修改配置项（gen_prompts_defaults.py 会读取），
  除占位符 {json} 外都可改
"""

from __future__ import annotations
import os
import json
from typing import Optional

DEEPSEEK_BASE = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
# deepseek-chat 定价（美元 / 百万 token，可用环境变量覆盖）
DS_IN_PER_M = float(os.environ.get("DEEPSEEK_IN_PER_M", "0.27"))
DS_OUT_PER_M = float(os.environ.get("DEEPSEEK_OUT_PER_M", "1.10"))

# 翻译引导提示词（可在「提示词修改」页编辑；{json} 为锁定占位符）
TRANSLATE_PROMPT = """你是资深广告本地化译者。下面【中文 JSON】是视频广告分析/生成流程产出的结构化提示词或卖点，将用于英文团队直接投喂 AI 视频生成模型。请把它翻译成地道、专业的英文，并严格遵守：
1. 只翻译「值」中的自然语言文本，中文→英文，用词精准专业；品牌名、型号、专有名词、数字、单位原样保留。
2. 所有「键名」原样保留，绝不翻译键名、不改结构、不增删字段、不改变数组顺序与元素个数。
3. 数值、布尔、时间戳、ID、文件名、以及用花括号包裹的占位符字段一律原样保留，不翻译。
4. 忠实原意、结构一一对应；行文精炼，控制长度，不做冗余扩写，保证下游生成效果。
5. 只输出翻译后的 JSON 本体，不要输出任何解释、Markdown 代码块或多余文字。

【中文 JSON】：
{json}"""


def _client(api_key: Optional[str] = None):
    from openai import OpenAI
    import httpx
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY（环境变量或调用时传入）")
    http_client = httpx.Client(timeout=180.0)
    return OpenAI(base_url=DEEPSEEK_BASE, api_key=key, http_client=http_client)


def cost_of(tin: int, tout: int) -> float:
    return tin / 1_000_000 * DS_IN_PER_M + tout / 1_000_000 * DS_OUT_PER_M


def pick_instr(override: Optional[str]) -> str:
    """合法覆盖优先（必须含 {json} 占位符），否则用默认。"""
    if isinstance(override, str) and "{json}" in override:
        return override
    return TRANSLATE_PROMPT


def translate_obj(obj, instr: str, client, model: str = None):
    """翻译一个 JSON 对象；返回 (英文对象 or None, tin, tout)。失败返回 (None, tin, tout)。"""
    model = model or DEEPSEEK_MODEL
    src = json.dumps(obj, ensure_ascii=False)
    prompt = instr.replace("{json}", src)
    tin = tout = 0
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        u = getattr(resp, "usage", None)
        if u:
            tin = int(getattr(u, "prompt_tokens", 0) or 0)
            tout = int(getattr(u, "completion_tokens", 0) or 0)
        txt = resp.choices[0].message.content or ""
        return json.loads(txt), tin, tout
    except Exception as e:
        print(f"[translate] 失败: {type(e).__name__}: {e}")
        return None, tin, tout
