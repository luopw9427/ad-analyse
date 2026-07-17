"""
Phase 1: 单条广告 → 视频级投喂 prompt
======================================

输入：一个广告的分析产物目录（含 shot_*.json + ad_aggregate.json）
输出：video_prompt.json —— 结构化 + 完整可投喂 AI 视频生成模型的长文本 prompt

CLI：
    python video_prompt_generator.py <analysis_dir> --product "..." [--pro]
"""

from __future__ import annotations
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ============================================================
# 一、Schema
# ============================================================

class 广告结构分类(BaseModel):
    主结构: Literal["PAS", "直给型", "Before-After", "UGC证言体", "演示型"] = Field(
        description="从 5 种经典结构里选一个最贴切的"
    )
    副结构: List[str] = Field(default_factory=list, description="可组合的副结构，如 ['直给型'] 或 ['UGC片段']")
    备注: str = Field(default="", description="一句话解释分类依据，≤30字")


class 镜头段(BaseModel):
    shot_id: str
    序号: int
    时段: str = Field(description="如 '0.0-2.0秒'")
    标题: str = Field(description="4-8 字概括，如 '开箱吸睛'、'动态跟随'")
    衔接前逻辑: Optional[str] = Field(
        default=None,
        description="从上一镜头如何切过来 + 情绪/节奏变化。第一个镜头必须为 null，其他必填"
    )
    衔接后逻辑: Optional[str] = Field(
        default=None,
        description="如何切到下一镜头 + 递进意图。最后一个镜头必须为 null，其他必填"
    )
    主要卖点: str = Field(description="4-8 字短标签")
    配合卖点: List[str] = Field(default_factory=list, description="0-3 个短标签")
    解决痛点: List[str] = Field(default_factory=list, description="0-3 个短标签")


class VideoPrompt(BaseModel):
    ad_id: str
    total_duration: float
    n_shots: int
    产品名: str

    广告结构分类: 广告结构分类
    video_prompt_text: str = Field(
        description="完整的可直接投喂 AI 视频生成模型的长文本 prompt，含全局约束段 + 逐镜头描述段"
    )
    整体流转逻辑: str = Field(
        description="描述整个广告的叙事弧线与情绪节奏，80-150 字"
    )
    商品卖点: List[str] = Field(default_factory=list, description="全广告去重后的短标签")
    解决痛点: List[str] = Field(default_factory=list, description="全广告去重后的短标签")
    镜头列表: List[镜头段] = Field(default_factory=list)


# ============================================================
# 二、内嵌广告结构分类参考（5 种）
# ============================================================

AD_STRUCTURE_REF = """
1. PAS（Problem–Agitate–Solution，痛点—放大—解决）：
   呈现痛点场景 → 放大痛感（后果 / 尴尬 / 反复失败） → 产品作为解决方案登场 → CTA。
   适合功能型商品，卖点是"解决某个明确痛点"。

2. 直给型（3 秒钩子 + 卖点连击 + CTA）：
   开场钩子直接抓人 → 快节奏堆 2-3 个卖点，一个镜头一个卖点 → 价格锚点 / 优惠 → CTA。
   没有故事、不绕弯，信息密度取胜。适合冲动购买/颜值展示型商品。

3. Before-After（对比型）：
   Before 惨状特写 → 产品介入的动作瞬间 → 转变过程 → After 惊艳结果 + CTA。
   逻辑是"眼见为实"，让画面替文案说话。适合效果强可视化商品（清洁 / 收纳 / 美妆）。

4. UGC 证言体（真人口播种草）：
   真人直面镜头抛强结论钩子 → 手持展示 + 使用过程 → 结果或轻对比 → 个人化推荐 + CTA。
   核心是"像朋友推荐，不像广告"。适合高客单信任型商品。

5. 演示/实验证明型（Demo / 极限测试）：
   悬念钩子 → 极限测试或核心功能演示过程 → 结果揭晓（爽点时刻） → 产品 + CTA。
   用一个不可辩驳的物理事实证明核心卖点。适合功能演示型商品。
"""


# ============================================================
# 三、主 Prompt
# ============================================================

PROMPT = """你是资深广告创意总监兼 AI 视频生成 prompt 工程师。

【任务】把广告分析结果整合成一段可直接投喂给 AI 视频生成模型（Sora / 可灵 / 智谱清言）
的完整视频脚本 prompt，并输出结构化 JSON。

【广告数据】

产品名：{product_name}
广告 id：{ad_id}
总时长：{total_duration:.2f} 秒
镜头数：{n_shots}

--- 每个镜头的详细分析（含分镜脚本级动作/展示/运镜）---
{shots_json}

--- 商业化整合结果（卖点/痛点/展示方式）---
{aggregate_json}

【广告结构分类参考（5 种，必须从中选一个作为主结构）】
{ad_structure_ref}

═══════════════════════════════════════════════════════

【任务 A：广告结构分类】

- 主结构：严格从 5 种里选一个（PAS / 直给型 / Before-After / UGC证言体 / 演示型）
- 副结构：如果混合了其他结构（如"演示 + 直给"），列出副结构名字；不混合就 []
- 备注：一句话解释判断依据（≤30 字）

═══════════════════════════════════════════════════════

【任务 B：生成 video_prompt_text（这是核心，最重要）】

必须严格遵循以下**格式模板**，输出一段可直接送 AI 视频生成模型的完整脚本长文本。

── 格式模板 ──

第一段：全局约束（一段完整描述，不分行）

  必须包含以下 10 个要素（顺序可微调，但都要提到）：
  1. 整体风格（如"高级广告风格"、"UGC 真实生活风格"）
  2. 画面质感（如"高清晰度、电影级光影、真实拍摄质感"）
  3. 模特类型（性别 / 年龄 / 地域面孔特征）—— ⚠️ **必须为美国面孔**，
     严禁输出亚洲面孔 / 欧洲面孔等其他描述
  4. 服装
  5. 音频要求（是否有旁白 / BGM / 音效）
  6. 参考图说明（用"图片1、图片2、图片3"指代产品图，说明产品在画面中的呈现要求）
  7. 构图（居中 / 是否留白 / 安全区）
  8. 播放速度（是否有慢动作）
  9. 字幕（有 / 无）
  10. BGM 结束时机

  示例（学这个语气和信息密度）：
  「高级广告风格，高清晰度，电影级光影，真实拍摄质感，服装保持准确，出镜模特为真实的
   美国女性面孔，身材符合美国大众审美。不要出现旁白或语音，只要 bgm。参考图是产品图
   图片1（大型秋季吊篮花，注意全程保留花篮的挂钩，要求全程产品的花体量放大百分之200，
   很大和饱满）。构图自然美观，视觉主体居中且饱满突出，无左右留白限制（不限制安全区）。
   全程为正常真实播放速度，无慢动作。全程无任何文字或字幕。视频结束时 bgm 也刚好结束。」

接下来：逐镜头描述，每个镜头一段（段与段之间用换行分开）

  格式：「镜头 N：标题（起始秒-结束秒） 景别与运镜。场景 + 人物动作 + 产品呈现 + 光影氛围。无字幕。」

  示例：
  「镜头 1：开箱吸睛（0-2.0秒） 全景转中景镜头。场景设定在阳光明媚的美式住宅前。
   一位穿着居家休闲服的美国女主，动作干脆利落地打开地上的两个快递纸箱，瞬间从中取出两个
   体量极其夸张的大型"秋季吊篮花"图片1。利用夸张的产品体积瞬间抓住观众眼球。画面居中，无字幕。」

  「镜头 2：动态跟随（2.0-4.0秒） 中景跟随镜头（Tracking Shot）。女主怀里抱着这两盆
   色彩浓郁的秋季花篮图片2，面带自然松弛的微笑，步履轻快地走向住宅的门廊（Porch）。
   电影级自然光在女主和鲜艳的秋季花朵上流转，极具美式真实生活气息与代入感。无字幕。」

── 硬性要求 ──

- 全局约束段必须包含以上 10 项要素，都要提到
- 参考图说明要具体到产品外观 / 关键特征（如挂钩、体量、颜色）
- 每个镜头段落必须包含：**镜头 N + 标题 + 时段 + 景别与运镜 + 场景 + 人物动作 + 产品呈现 + 无字幕说明**
- 时段严格按 shot 顺序累积，从 0 开始
- 参考"图片1 / 图片2 / 图片3"根据镜头需要引用
- **人物动作/产品展示/景别** 都从广告分析里的 note 提炼具体细节，不要泛泛而谈
- 无字幕要在每个镜头结尾单独强调"无字幕"

═══════════════════════════════════════════════════════

【任务 C：镜头列表结构化数据】

每个镜头填：序号 / 时段 / 标题 / 衔接前逻辑 / 衔接后逻辑 / 主要卖点 / 配合卖点 / 解决痛点

- **衔接前逻辑**：第一个镜头必须为 null，其他镜头必填。描述"从上一镜头切过来的方式 + 情绪/节奏变化"
  示例："硬切，从口播悬念转到户外全景，节奏由停顿转为快节奏"
- **衔接后逻辑**：最后一个镜头必须为 null，其他镜头必填。描述"如何切到下一镜头 + 递进意图"
  示例："平滑过渡到微距特写，把注意力从场景收拢到产品细节"
- **主要卖点**：一个 4-8 字短标签，如"体量夸张"
- **配合卖点**：0-3 个短标签
- **解决痛点**：0-3 个短标签

═══════════════════════════════════════════════════════

【任务 D：整体流转逻辑】
80-150 字，描述整个广告从开头到结尾的叙事弧线与情绪节奏。
示例："开场用夸张体量的开箱瞬间抓住注意，紧接中景跟随建立生活代入感，
       随后近景特写切入产品细节，微距爆发式呈现产品核心卖点，
       最后拉远全景定格产品应用场景，形成'抓眼-代入-细节-爆点-升华'的情绪弧线。"

═══════════════════════════════════════════════════════

【任务 E：商品卖点 / 解决痛点】
从 ad_aggregate 里去重合并出短标签数组，可以少量补充。

═══════════════════════════════════════════════════════

严格按下面 JSON schema 输出，不要 markdown 代码块，不要多余文字：

{schema_json}
"""


# ============================================================
# 四、OpenRouter 客户端
# ============================================================

def _make_client(api_key: Optional[str] = None):
    from openai import OpenAI
    import httpx
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("缺少 OPENROUTER_API_KEY")
    http_client = httpx.Client(timeout=300.0)
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        http_client=http_client,
        default_headers={
            "HTTP-Referer": "https://local.video-prompt-gen",
            "X-Title": "Video Prompt Generator",
        },
    )


def _repair_llm_json(raw: str) -> str:
    import re
    raw = re.sub(r'(,\s*)[a-zA-Z_][a-zA-Z_0-9]*"\s*:\s*', r'\1', raw)
    raw = re.sub(r'^\s*[a-zA-Z_][a-zA-Z_0-9]*"\s*:\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r':\s*"null"\s*(,|\n|\})', r': null\1', raw)
    raw = re.sub(r',(\s*[\]\}])', r'\1', raw)
    return raw


# ============================================================
# 五、主函数
# ============================================================

def generate_video_prompt(
    analysis_dir: Path,
    product_name: str,
    output_path: Optional[Path] = None,
    model: str = "google/gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> VideoPrompt:
    """
    读取分析目录里的所有 shot_*.json + ad_aggregate.json，
    调 Gemini 生成 video_prompt.json。
    """
    analysis_dir = Path(analysis_dir)
    if not analysis_dir.is_dir():
        raise RuntimeError(f"目录不存在: {analysis_dir}")

    shot_files = sorted(analysis_dir.glob("shot_*.json"))
    if not shot_files:
        raise RuntimeError(f"未找到 shot_*.json in {analysis_dir}")

    aggregate_file = analysis_dir / "ad_aggregate.json"
    if not aggregate_file.exists():
        raise RuntimeError(f"未找到 ad_aggregate.json in {analysis_dir}")

    shots = []
    for fp in shot_files:
        with open(fp, "r", encoding="utf-8") as f:
            shots.append(json.load(f))

    with open(aggregate_file, "r", encoding="utf-8") as f:
        aggregate = json.load(f)

    ad_id = aggregate.get("ad_id") or analysis_dir.parent.name
    total_duration = float(aggregate.get("total_duration", 0.0))
    n_shots = int(aggregate.get("n_shots", len(shots)))

    # 精简每个 shot 只送有用字段（保留完整 note 供 LLM 使用）
    slim_shots = []
    for s in shots:
        chains = s.get("action_chains", {}) or {}
        slim_shots.append({
            "shot_id": s.get("shot_id"),
            "start_ts": s.get("start_ts"),
            "end_ts": s.get("end_ts"),
            "shot_intent": s.get("shot_intent_one_liner"),
            "human_actions": [
                {
                    "verb": a.get("verb"),
                    "body_part": a.get("body_part"),
                    "direction": a.get("direction"),
                    "target": a.get("target"),
                    "force": a.get("force"),
                    "note": a.get("note"),
                }
                for a in chains.get("human", [])
            ],
            "product_presentation": [
                {
                    "type": b.get("type"),
                    "focus": b.get("focus"),
                    "technique": b.get("technique"),
                    "note": b.get("note"),
                }
                for b in chains.get("product", [])
            ],
            "camera": [c.get("motion") for c in chains.get("camera", [])],
            "overlay": [o.get("text", "") for o in chains.get("overlay", [])],
            "selling_points": [
                {"name": sp.get("name"), "feature": sp.get("feature"), "benefit": sp.get("benefit")}
                for sp in s.get("selling_points", [])
            ],
        })

    prompt_text = PROMPT.format(
        product_name=product_name,
        ad_id=ad_id,
        total_duration=total_duration,
        n_shots=n_shots,
        shots_json=json.dumps(slim_shots, ensure_ascii=False, indent=2),
        aggregate_json=json.dumps(aggregate, ensure_ascii=False, indent=2),
        ad_structure_ref=AD_STRUCTURE_REF,
        schema_json=json.dumps(VideoPrompt.model_json_schema(), ensure_ascii=False, indent=2),
    )

    client = _make_client(api_key)
    is_pro = "pro" in model.lower()
    req_kwargs = (
        {"max_tokens": 32000, "extra_body": {"reasoning": {"effort": "medium", "exclude": True}}}
        if is_pro else
        {"max_tokens": 16000, "extra_body": {"reasoning": {"max_tokens": 0, "exclude": True}}}
    )

    print(f"    [VideoPrompt] 用 {model} 生成视频级 prompt...")

    max_retries = 5
    resp = None
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                response_format={"type": "json_object"},
                temperature=0.3,
                **req_kwargs,
            )
            usage = getattr(resp, "usage", None)
            if usage:
                print(f"    [VideoPrompt] usage: prompt={usage.prompt_tokens} / completion={usage.completion_tokens}")
            finish = resp.choices[0].finish_reason
            raw_check = (resp.choices[0].message.content or "").strip()
            token_zero = usage and usage.completion_tokens == 0
            hard_fail = (finish == "error" and token_zero) or len(raw_check) < 30
            if hard_fail and attempt < max_retries:
                wait = min(5 * attempt, 30)
                print(f"    [VideoPrompt] [重试 {attempt}/{max_retries}] finish={finish}, len={len(raw_check)}, 等 {wait}s")
                time.sleep(wait)
                continue
            break
        except Exception as e:
            last_err = e
            wait = min(5 * attempt, 30)
            print(f"    [VideoPrompt] [重试 {attempt}/{max_retries}] 异常: {e}，等 {wait}s")
            if attempt < max_retries:
                time.sleep(wait)
            continue

    if resp is None:
        raise RuntimeError(f"[VideoPrompt] 重试 {max_retries} 次仍失败: {last_err}")

    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()
    raw = _repair_llm_json(raw)

    try:
        result = VideoPrompt.model_validate_json(raw)
    except Exception as e1:
        # 尝试 json_repair
        try:
            import json_repair  # type: ignore
            fixed = json_repair.loads(raw)
            result = VideoPrompt.model_validate(fixed)
            print(f"    [VideoPrompt] json_repair 修复成功")
        except Exception:
            debug_path = analysis_dir / f"debug_{ad_id}_video_prompt_raw.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"--- 首次错误 ---\n{e1}\n\n--- RAW ---\n{raw}")
            raise RuntimeError(f"[VideoPrompt] JSON 解析失败，原始响应存到 {debug_path}")

    # 硬填入元信息
    result.ad_id = ad_id
    result.total_duration = round(total_duration, 3)
    result.n_shots = n_shots
    result.产品名 = product_name

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        print(f"    [VideoPrompt] 已保存 → {output_path}")

    return result


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从广告分析产物生成视频级投喂 prompt")
    parser.add_argument("analysis_dir", help="分析目录（含 shot_*.json + ad_aggregate.json）")
    parser.add_argument("--product", required=True, help="产品名")
    parser.add_argument("--pro", action="store_true", help="用 gemini-2.5-pro")
    parser.add_argument("--api-key", help="OpenRouter API Key")
    parser.add_argument("--output", help="输出路径（默认 <analysis_dir>/video_prompt.json）")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_path = Path(args.output) if args.output else analysis_dir / "video_prompt.json"
    model = "google/gemini-2.5-pro" if args.pro else "google/gemini-2.5-flash"

    generate_video_prompt(
        analysis_dir=analysis_dir,
        product_name=args.product,
        output_path=output_path,
        model=model,
        api_key=args.api_key,
    )
