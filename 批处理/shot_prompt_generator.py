"""
Phase 1.5: 单条广告 → 镜头级 prompt 素材库
============================================

输入：某条广告的 video_prompt.json
输出：shot_prompts.json —— 至少 7 条独立的、可直接投喂 AI 视频生成模型的
      单镜头 prompt（每条围绕不同的卖点/痛点/展示方式）

CLI：
    python shot_prompt_generator.py <video_prompt.json 路径> [--min-count 7] [--pro]
"""

from __future__ import annotations
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

# 复用 video_prompt_generator 的客户端和修复工具
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_prompt_generator import _make_client, _repair_llm_json


# ============================================================
# 一、Schema（精简版）
# ============================================================

class 镜头级Prompt(BaseModel):
    id: str = Field(description="sp_01, sp_02, ...")
    标题: str = Field(description="4-8 字概括，如 '拆袋绽放特写'、'门廊定格'")
    卖点: List[str] = Field(default_factory=list, description="1-3 个短标签")
    痛点: List[str] = Field(default_factory=list, description="0-3 个短标签")
    prompt: str = Field(
        description="完整可直接投喂 AI 视频生成模型的单镜头 prompt 长文本，含全局约束段 + 详细镜头段"
    )


class ShotPromptLibrary(BaseModel):
    ad_id: str
    产品名: str
    source_video_prompt: str = Field(description="来源 video_prompt.json 的相对路径")
    镜头级prompt列表: List[镜头级Prompt] = Field(min_length=7)


# ============================================================
# 二、Prompt 模板
# ============================================================

PROMPT = """你是资深广告分镜师兼 AI 视频生成 prompt 工程师。

【任务】基于以下广告的完整分析，产出至少 {min_count} 条**独立可用**的镜头级 prompt 素材。
每条都可以单独投喂给 AI 视频生成模型（Sora / 可灵 / 智谱清言）生成一个完整视频镜头。

═══ 输入：广告的 video_prompt.json ═══

{video_prompt_json}

═══════════════════════════════════════════════════════

【任务要求】

1. **数量**：至少 {min_count} 条（可以更多）

2. **来源策略**：
   - 部分来自原广告镜头的**提取**（把原广告的某个镜头独立成完整 prompt）
   - 部分来自**扩充**（围绕某个卖点/痛点/展示方式，设计原广告没有但符合调性的新镜头）
   - 覆盖所有主要卖点和痛点

3. **每条 prompt 的结构**（严格模板 —— **两段式，用换行分隔**，不要引入子分段）：

   ── 第一段：全局约束段 ──
   一段完整描述，含风格 / 光影 / 模特 / 服装 / 音频 / 参考图（用"图片1"）
   / 构图 / 速度 / 字幕 / BGM 结束时机。
   ⚠️ **模特必须为美国面孔**（真实的美国女性面孔，身材符合美国大众审美），
   **严禁**输出亚洲面孔 / 欧洲面孔等其他描述。

   ── 换行 ──

   ── 第二段：镜头段 ──
   一段**连贯自然**的描述（不要用 [段1][段2] 这种分块，就是一整段文字），
   **必须 250-400 字**（是全局约束段的 2 倍以上），
   **必须含至少 8 个可量化词**。

   格式：「镜头：标题 景别与运镜（含技术参数）。详细描述动作过程与画面细节。图片1。无字幕。」

   ⚠️ 描述必须涵盖（在一段文字里自然融合）：
   - 谁在画面里 / 从哪进入 / 位置和姿态
   - 产品初始状态、尺寸、颜色、材质
   - 动作的方向、力度、速度、持续时长
   - 动作过程中产品的变化
   - 光位、光型、氛围
   - 结束时的定格画面

   ✅ 正例（学这个语气，一整段连贯，含大量数字）：
   「镜头：撕膜点钻过程。俯视中景，焦距 85mm，光圈 f/2.8。女性双手从画面左侧边缘伸入，
   指尖轻捏水钻贴画左上角的透明保护膜边缘，缓慢向右撕开约四分之一的面积，
   露出下方带有胶层的图案区域，持续约 2.5 秒。女性右手持点钻笔，笔尖沾取一颗水钻，
   精准地将水钻放置在贴画图案的指定位置上，轻按使其固定，产品占画面 75%。
   镜头保持稳定，展示女性重复沾取水钻并放置的动作，强调操作的流畅和简便。图片1。无字幕。」

   ❌ 反例（不要输出）：
   - 太短：「近景特写。女主拆开保护袋，无字幕。」
   - 分块：「[段1] xxx。[段2] xxx。」
   - 没数字：「镜头稳定，女性动作流畅，展示产品优秀。」

   ★ 可量化词至少 **8 个**，从以下维度选：
   - 焦距（85mm / 100mm / 24mm 广角）
   - 光圈（f/2.8 / f/4 / f/8）
   - 机位角度（俯视 45 度 / 平视 / 仰视 15 度）
   - 画面占比（产品占 75%）
   - 尺寸/距离（直径 20cm / 40cm×50cm）
   - 时长（持续 2.5 秒 / 约 3 秒）
   - 速度（每秒 15cm 匀速 / 缓慢下探）
   - 光位（左侧 45 度自然侧光 / 顶光偏左）
   - 数量（6 片 / 一颗水钻 / 四分之一面积）
   - 材质细节（PE 塑料 / 果冻质感胶层 / 亚克力水钻）
   - 颜色（浅粉 / 蓝紫渐变）
   - 动作方向（向左下 45 度斜拉 / 由内向外画圆）

4. **每条 prompt 围绕不同角度**：
   - 微距材质特写（展示仿真度）
   - 场景应用（展示使用后效果）
   - 开箱/拆袋（展示便捷性）
   - 对比展示（展示体量、颜色多样）
   - 挂载/摆放动作（展示无需组装）
   - 生活代入场景（展示情感价值）
   - 定格终镜（展示最终氛围）

5. **每条 prompt 的结构化字段**：
   - id：sp_01, sp_02, ...
   - 标题：4-8 字概括
   - 卖点：1-3 个短标签
   - 痛点：0-3 个短标签
   - prompt：完整可投喂长文本（含全局约束段 + 详细镜头段）

═══════════════════════════════════════════════════════

严格按下面 JSON schema 输出，不要 markdown 代码块，不要多余文字：

{schema_json}
"""


# ============================================================
# 三、主函数
# ============================================================

def _usage_cost(usage) -> float:
    """OpenRouter 每次响应都会带真实扣费金额（usage.cost）。取不到则返回 0。"""
    if usage is None:
        return 0.0
    c = getattr(usage, "cost", None)
    if c is None:
        extra = getattr(usage, "model_extra", None) or {}
        c = extra.get("cost")
    try:
        return float(c or 0.0)
    except (TypeError, ValueError):
        return 0.0


def generate_shot_prompts(
    video_prompt_path: Path,
    output_path: Optional[Path] = None,
    min_count: int = 7,
    model: str = "google/gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> ShotPromptLibrary:
    """
    从广告 video_prompt.json 生成至少 min_count 条独立镜头级 prompt。
    """
    video_prompt_path = Path(video_prompt_path)
    if not video_prompt_path.is_file():
        raise RuntimeError(f"video_prompt.json 不存在: {video_prompt_path}")

    with open(video_prompt_path, "r", encoding="utf-8") as f:
        vp = json.load(f)

    ad_id = vp.get("ad_id", "unknown")
    product_name = vp.get("产品名", "未知产品")

    prompt_text = PROMPT.format(
        min_count=min_count,
        video_prompt_json=json.dumps(vp, ensure_ascii=False, indent=2),
        schema_json=json.dumps(ShotPromptLibrary.model_json_schema(), ensure_ascii=False, indent=2),
    )

    client = _make_client(api_key)
    is_pro = "pro" in model.lower()
    req_kwargs = (
        {"max_tokens": 32000, "extra_body": {"reasoning": {"effort": "medium", "exclude": True}}}
        if is_pro else
        {"max_tokens": 20000, "extra_body": {"reasoning": {"max_tokens": 0, "exclude": True}}}
    )

    print(f"    [ShotPrompts] 用 {model} 生成 ≥{min_count} 条独立镜头级 prompt...")

    max_retries = 5
    resp = None
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                response_format={"type": "json_object"},
                temperature=0.4,  # 稍高鼓励多样性
                **req_kwargs,
            )
            usage = getattr(resp, "usage", None)
            if usage:
                print(f"    [ShotPrompts] usage: prompt={usage.prompt_tokens} / completion={usage.completion_tokens} / cost=${_usage_cost(usage):.6f}")
            finish = resp.choices[0].finish_reason
            raw_check = (resp.choices[0].message.content or "").strip()
            token_zero = usage and usage.completion_tokens == 0
            hard_fail = (finish == "error" and token_zero) or len(raw_check) < 30
            if hard_fail and attempt < max_retries:
                wait = min(5 * attempt, 30)
                print(f"    [ShotPrompts] [重试 {attempt}/{max_retries}] finish={finish}, len={len(raw_check)}, 等 {wait}s")
                time.sleep(wait)
                continue
            break
        except Exception as e:
            last_err = e
            wait = min(5 * attempt, 30)
            print(f"    [ShotPrompts] [重试 {attempt}/{max_retries}] 异常: {e}，等 {wait}s")
            if attempt < max_retries:
                time.sleep(wait)
            continue

    if resp is None:
        raise RuntimeError(f"[ShotPrompts] 重试 {max_retries} 次仍失败: {last_err}")

    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()
    raw = _repair_llm_json(raw)

    try:
        result = ShotPromptLibrary.model_validate_json(raw)
    except Exception as e1:
        try:
            import json_repair  # type: ignore
            fixed = json_repair.loads(raw)
            result = ShotPromptLibrary.model_validate(fixed)
            print(f"    [ShotPrompts] json_repair 修复成功")
        except Exception:
            debug_path = video_prompt_path.parent / f"debug_{ad_id}_shot_prompts_raw.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"--- 首次错误 ---\n{e1}\n\n--- RAW ---\n{raw}")
            raise RuntimeError(f"[ShotPrompts] JSON 解析失败，原始响应存到 {debug_path}")

    # 硬填入元信息
    result.ad_id = ad_id
    result.产品名 = product_name
    # 只存文件名，避免把临时/绝对路径泄漏进产物
    result.source_video_prompt = Path(video_prompt_path).name

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        n = len(result.镜头级prompt列表)
        print(f"    [ShotPrompts] {n} 条独立镜头 prompt → {output_path}")

    return result


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="从 video_prompt.json 生成独立镜头级 prompt 素材库"
    )
    parser.add_argument("video_prompt", help="video_prompt.json 文件路径")
    parser.add_argument("--min-count", type=int, default=7, help="至少生成几条（默认 7）")
    parser.add_argument("--pro", action="store_true", help="用 gemini-2.5-pro")
    parser.add_argument("--api-key", help="OpenRouter API Key")
    parser.add_argument("--output", help="输出路径（默认 <同目录>/shot_prompts.json）")
    args = parser.parse_args()

    vp_path = Path(args.video_prompt)
    output_path = Path(args.output) if args.output else vp_path.parent / "shot_prompts.json"
    model = "google/gemini-2.5-pro" if args.pro else "google/gemini-2.5-flash"

    generate_shot_prompts(
        video_prompt_path=vp_path,
        output_path=output_path,
        min_count=args.min_count,
        model=model,
        api_key=args.api_key,
    )
