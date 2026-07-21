"""
Phase 1.6: 品类级共性镜头 prompt 素材库
========================================

从批处理的所有 video_prompt.json 中归纳共性（共通卖点/痛点/展示套路/镜头语言），
生成一套**跨广告通用的品类级镜头 prompt 素材库**——可作为拍同品类新广告时的标准镜头库。

CLI：
    python common_shot_prompt_generator.py <根目录> --product "..." [--pipeline merged/split] [--min-count 7] [--pro]
"""

from __future__ import annotations
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_prompt_generator import _make_client, _repair_llm_json


# ============================================================
# 一、Schema
# ============================================================

class 共性镜头Prompt(BaseModel):
    id: str = Field(description="cp_01, cp_02, ...")
    标题: str = Field(description="4-8 字概括")
    卖点: List[str] = Field(default_factory=list, description="1-3 个共性卖点短标签")
    痛点: List[str] = Field(default_factory=list, description="0-3 个共性痛点短标签")
    prompt: str = Field(description="完整可直接投喂 AI 视频生成模型的单镜头 prompt 长文本")


class CommonShotPromptLibrary(BaseModel):
    品类: str = Field(description="一句话概括，如 '户外人造仿真花'")
    样本数: int
    样本ad_ids: List[str]
    商品共性卖点: List[str] = Field(description="多条广告去重后的高频卖点短标签，5-10 条")
    共性痛点: List[str] = Field(description="多条广告去重后的高频痛点短标签，3-8 条")
    共性镜头prompt列表: List[共性镜头Prompt] = Field(min_length=7)


# ============================================================
# 二、Prompt 模板
# ============================================================

PROMPT = """你是资深广告分镜师兼 AI 视频生成 prompt 工程师。

【任务】基于以下 {n_ads} 条同品类广告的完整分析，归纳**共性**（共通高频卖点/痛点/
展示套路/开箱套路/镜头语言），设计一套**跨广告通用的品类级镜头 prompt 素材库**——
可作为拍这个品类新广告时的"标准镜头库"，每条都是完整可直接投喂 AI 视频生成模型的
独立单镜头 prompt。

═══ 输入：{n_ads} 条广告的完整分析 ═══

{ads_json}

═══════════════════════════════════════════════════════

【任务要求】

1. **数量**：至少 {min_count} 条独立镜头 prompt

2. **归纳共性（不要单纯罗列某一条广告的细节）**：
   - 从多条广告中提炼**共同出现的高频卖点/痛点/开箱套路/镜头语言**
   - 关注这个品类广告惯用的：
     • 钩子方式（如"体量反差"、"揭袋惊喜"、"人物直面镜头"）
     • 核心卖点话术（如"3年不凋谢"、"免打理"、"送礼首选"）
     • 结尾套路（如"多户成果轮播"、"稀缺 CTA"、"生活场景定格"）
     • 镜头语言（微距材质特写、生活代入中景、俯视全景等）
     • 音乐/口播套路

3. **每条 prompt 反映品类共性**：
   - 不局限于任何一条具体广告的具体画面细节
   - 而是"这个品类的广告普遍怎么拍"的通用镜头
   - 每条围绕一个**共性卖点 / 痛点 / 展示套路**设计

4. **每条 prompt 结构**（严格模板 —— **两段式，用换行分隔**，不要引入子分段）：

   ── 第一段：全局约束段 ──
   一段完整描述，包含风格 / 光影 / 模特 / 服装 / 音频 / 参考图（用"图片1"）
   / 构图 / 速度 / 字幕 / BGM 结束时机。
   ⚠️ **模特必须为美国面孔**（真实的美国女性面孔，身材符合美国大众审美），
   **严禁**输出亚洲面孔 / 欧洲面孔等其他描述。

   示例：
   「高级广告风格，高清晰度，电影级光影，真实拍摄质感，服装保持准确，出镜模特为
    真实的美国女性面孔，身材符合美国大众审美。不要出现旁白或语音，只要 bgm。
    参考图是产品图图片1（产品外观 / 关键特征描述）。构图自然美观，视觉主体居中
    且饱满突出，无左右留白限制。全程为正常真实播放速度，无慢动作。全程无任何
    文字或字幕。视频结束时 bgm 也刚好结束。」

   ── 换行 ──

   ── 第二段：镜头段 ──
   一段**连贯自然**的描述（不要用 [段1][段2] 这种分块，就是一整段文字），
   **必须 250-400 字**（是全局约束段的 2 倍以上长度），
   **必须含至少 8 个可量化词**。

   格式：「镜头：标题 景别与运镜（含技术参数）。详细描述动作过程与画面细节。图片1。无字幕。」

   ⚠️ 描述必须涵盖以下要素（在一段文字里自然融合，不分小节）：
   - 谁在画面里 / 从哪进入 / 位置和姿态
   - 产品初始状态、尺寸、颜色、材质细节
   - 动作的方向、力度、速度、持续时长（有起止时间戳更好）
   - 动作过程中产品的变化（如撕开面积、绽放程度、颜色渐变）
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
   - 尺寸/距离（直径 20cm / 40cm×50cm / 撕开 25cm 幅度）
   - 时长（持续 2.5 秒 / 约 3 秒）
   - 速度（每秒 15cm 匀速 / 缓慢下探）
   - 光位（左侧 45 度自然侧光 / 顶光偏左）
   - 数量（6 片花瓣 / 一颗水钻 / 四分之一面积）
   - 材质细节（PE 塑料 / 果冻质感胶层 / 亚克力水钻）
   - 颜色（浅粉 / 蓝紫渐变 / 暖调木纹）
   - 动作方向（向左下 45 度斜拉 / 由内向外画圆）

5. **prompt 输出字段（精简 5 项）**：
   - id：cp_01, cp_02, ...（cp = common prompt）
   - 标题：4-8 字概括
   - 卖点：1-3 个**共性卖点**短标签
   - 痛点：0-3 个**共性痛点**短标签
   - prompt：完整可投喂长文本

6. **顶层字段**：
   - 品类：一句话概括
   - 样本数、样本ad_ids
   - 商品共性卖点：去重后的高频卖点（5-10 条）
   - 共性痛点：去重后的高频痛点（3-8 条）
   - 共性镜头prompt列表：≥ {min_count} 条

7. **覆盖不同镜头类型**（至少涵盖以下类别中的 5 种）：
   - 开箱瞬间（体量反差 / 揭袋惊喜）
   - 微距材质特写（展示仿真度 / 材质细节）
   - 人物动作演示（挂载 / 摆放 / 拆包）
   - 生活场景代入（门廊 / 桌面 / 户外）
   - 多户/多角度成果轮播
   - 对比展示（before/after 或 多色 SKU 快切）
   - 终镜定格（稀缺 CTA 或情感升华）

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


def generate_common_shot_prompts(
    root: Path,
    product_name: str,
    output_path: Optional[Path] = None,
    min_count: int = 7,
    pipeline: str = "merged",
    model: str = "google/gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> CommonShotPromptLibrary:
    """
    读取根目录下所有广告的 video_prompt.json，汇总生成品类级共性镜头 prompt 素材库。
    """
    root = Path(root)
    if not root.is_dir():
        raise RuntimeError(f"根目录不存在: {root}")

    subdir = "analysis_output" if pipeline == "merged" else "analysis_output_split"

    ads_data = []
    for ad_folder in sorted(root.iterdir()):
        if not ad_folder.is_dir():
            continue
        analysis_dir = ad_folder / subdir
        vp_path = analysis_dir / "video_prompt.json"
        if not vp_path.exists():
            continue
        with open(vp_path, "r", encoding="utf-8") as f:
            vp = json.load(f)
        ads_data.append({
            "ad_id": vp.get("ad_id", ad_folder.name),
            "total_duration": vp.get("total_duration"),
            "n_shots": vp.get("n_shots"),
            "广告结构分类": vp.get("广告结构分类"),
            "商品卖点": vp.get("商品卖点"),
            "解决痛点": vp.get("解决痛点"),
            "整体流转逻辑": vp.get("整体流转逻辑"),
            "镜头列表": [
                {
                    "序号": s.get("序号"),
                    "标题": s.get("标题"),
                    "主要卖点": s.get("主要卖点"),
                    "配合卖点": s.get("配合卖点"),
                    "解决痛点": s.get("解决痛点"),
                }
                for s in vp.get("镜头列表", [])
            ],
            "video_prompt_text_摘要": (vp.get("video_prompt_text", "") or "")[:800],
        })

    if not ads_data:
        raise RuntimeError(f"未在 {root} 找到任何 video_prompt.json（是否已跑完批处理？）")

    ad_ids = [a["ad_id"] for a in ads_data]

    prompt_text = PROMPT.format(
        n_ads=len(ads_data),
        min_count=min_count,
        ads_json=json.dumps(ads_data, ensure_ascii=False, indent=2),
        schema_json=json.dumps(CommonShotPromptLibrary.model_json_schema(), ensure_ascii=False, indent=2),
    )

    client = _make_client(api_key)
    is_pro = "pro" in model.lower()
    req_kwargs = (
        {"max_tokens": 32000, "extra_body": {"reasoning": {"effort": "medium", "exclude": True}}}
        if is_pro else
        {"max_tokens": 24000, "extra_body": {"reasoning": {"max_tokens": 0, "exclude": True}}}
    )

    print(f"[CommonShotPrompts] 用 {model} 从 {len(ads_data)} 条广告归纳共性，生成 ≥{min_count} 条品类级镜头 prompt...")

    max_retries = 5
    resp = None
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                response_format={"type": "json_object"},
                temperature=0.4,
                **req_kwargs,
            )
            usage = getattr(resp, "usage", None)
            if usage:
                print(f"[CommonShotPrompts] usage: prompt={usage.prompt_tokens} / completion={usage.completion_tokens} / cost=${_usage_cost(usage):.6f}")
            finish = resp.choices[0].finish_reason
            raw_check = (resp.choices[0].message.content or "").strip()
            token_zero = usage and usage.completion_tokens == 0
            hard_fail = (finish == "error" and token_zero) or len(raw_check) < 30
            if hard_fail and attempt < max_retries:
                wait = min(5 * attempt, 30)
                print(f"[CommonShotPrompts] [重试 {attempt}/{max_retries}] finish={finish}, len={len(raw_check)}, 等 {wait}s")
                time.sleep(wait)
                continue
            break
        except Exception as e:
            last_err = e
            wait = min(5 * attempt, 30)
            print(f"[CommonShotPrompts] [重试 {attempt}/{max_retries}] 异常: {e}，等 {wait}s")
            if attempt < max_retries:
                time.sleep(wait)
            continue

    if resp is None:
        raise RuntimeError(f"[CommonShotPrompts] 重试 {max_retries} 次仍失败: {last_err}")

    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()
    raw = _repair_llm_json(raw)

    try:
        result = CommonShotPromptLibrary.model_validate_json(raw)
    except Exception as e1:
        try:
            import json_repair  # type: ignore
            fixed = json_repair.loads(raw)
            result = CommonShotPromptLibrary.model_validate(fixed)
            print(f"[CommonShotPrompts] json_repair 修复成功")
        except Exception:
            debug_path = root / "debug_common_shot_prompts_raw.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"--- 首次错误 ---\n{e1}\n\n--- RAW ---\n{raw}")
            raise RuntimeError(f"[CommonShotPrompts] JSON 解析失败，原始响应存到 {debug_path}")

    result.样本数 = len(ads_data)
    result.样本ad_ids = ad_ids

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        n = len(result.共性镜头prompt列表)
        print(f"[CommonShotPrompts] {n} 条品类级共性镜头 prompt → {output_path}")

    return result


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从批处理产物汇总生成品类级共性镜头 prompt")
    parser.add_argument("root", help="根目录（下有各广告子文件夹）")
    parser.add_argument("--product", required=True, help="产品名")
    parser.add_argument("--pipeline", choices=["merged", "split"], default="merged")
    parser.add_argument("--min-count", type=int, default=7)
    parser.add_argument("--pro", action="store_true")
    parser.add_argument("--api-key")
    parser.add_argument("--output", help="默认 <根目录>/共性镜头prompt.json")
    args = parser.parse_args()

    root = Path(args.root)
    output_path = Path(args.output) if args.output else root / "共性镜头prompt.json"
    model = "google/gemini-2.5-pro" if args.pro else "google/gemini-2.5-flash"

    generate_common_shot_prompts(
        root=root,
        product_name=args.product,
        output_path=output_path,
        min_count=args.min_count,
        pipeline=args.pipeline,
        model=model,
        api_key=args.api_key,
    )
