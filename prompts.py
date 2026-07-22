"""
提示词配置（可覆盖）
====================

10 个分析环节的提示词默认在各自模块里；运行时若设置了环境变量 PROMPTS_FILE
（指向一个 {key: 提示词文本} 的 JSON），则用其中的覆盖版本。
覆盖前会校验：该环节必需的 {占位符} 必须都在，否则回退默认——绝不让分析崩。

默认文本 + 必需占位符 存在 prompts_defaults.json（由 gen_prompts_defaults.py 生成）。
"""

import os
import json
import string

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEF_FILE = os.path.join(_DIR, "prompts_defaults.json")


def load_defaults() -> dict:
    try:
        with open(_DEF_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def required_vars(key: str) -> set:
    d = load_defaults().get(key) or {}
    return set(d.get("required", []))


def _overrides() -> dict:
    fp = os.environ.get("PROMPTS_FILE")
    if not fp or not os.path.exists(fp):
        return {}
    try:
        with open(fp, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def fields_in(text: str):
    """提取模板里的占位符字段名集合。"""
    try:
        return {fn for _, fn, _, _ in string.Formatter().parse(text) if fn}
    except Exception:
        return None


def pick(key: str, default: str) -> str:
    """返回该环节运行时提示词：合法覆盖优先，否则默认。"""
    ov = _overrides().get(key)
    if not ov or not isinstance(ov, str):
        return default
    need = required_vars(key)
    have = fields_in(ov)
    if have is None or not need.issubset(have):
        return default
    return ov
