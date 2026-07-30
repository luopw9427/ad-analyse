"""
Phase 1: 单条广告 → 视频级投喂 prompt
======================================

输入：一个广告的分析产物目录（含 shot_*.json + ad_aggregate.json）
输出：video_prompt.json —— 结构化 + 完整可投喂 AI 视频生成模型的长文本 prompt

CLI：
    python video_prompt_generator.py <analysis_dir> --product "..." [--pro]
"""

from __future__ import annotations
# ── 提示词可覆盖（PROMPTS_FILE 存在时用覆盖版，否则用下方默认常量）──
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import prompts as _pcfg
    def _prompt(key, default):
        return _pcfg.pick(key, default)
except Exception:
    def _prompt(key, default):
        return default

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

输出一段可直接送 AI 视频生成模型的完整脚本长文本，由「全局约束段 + 逐镜头段」组成。
**核心原则：以镜头语言为主、细节要足。** 保留能指导画面生成的具体视觉信息
（景别、运镜、主体位置与动作过程、产品外观与状态变化、光位光型、材质细节、速度、字幕），
把画面写清楚写具体——**宁可略详细，也不要简到无法据此生成**。
只删掉三类冗余：① 目的/解说词（"展示…以突出…""强调…"）；② 氛围/情绪/评价（"营造宁静氛围""和谐有层次""珍视"）；
③ 对参考图内容的重复复述（改用"与图片X一致"）。画面本身的动作、细节、光影**不要删**。

── 全局约束段（一段完整描述，不分行）──

  必须包含以下要素（顺序可微调）：
  1. 整体风格（如"高级广告风格"）
  2. 画面质感（如"高清晰度、电影级光影、真实拍摄质感"）
  3. 模特类型 —— ⚠️ **必须为美国面孔**，严禁亚洲/欧洲等其他描述
  4. 服装
  5. 音频（是否旁白 / BGM / 音效）
  6. 参考图定义（用"图片1、图片2、图片3"指代产品图，**在这里一次性写清每张图是什么**；
     后文镜头里不再重复描述图片内容）
  7. 构图（居中 / 留白 / 安全区）
  8. 播放速度（是否慢动作 / 快节奏）
  9. 画面文字总基调（**忠实原广告**：若原广告用了花字/字幕/slogan，则说明整体有动态英文花字，
     具体文案写在对应镜头里；若原广告无文字，则"全程无任何文字或字幕"）
  10. BGM 结束时机

  示例（学这个语气和信息密度）：
  「高级广告风格，高清晰度，电影级光影，真实拍摄质感，服装保持准确，出镜模特为真实的
   美国女性面孔，身材符合美国大众审美。不要出现旁白或语音，只要 bgm。参考图是产品图
   图片1（完成的海洋主题贴片钻石画，由多种大小、颜色和形状的钻石贴片组成，闪耀立体），
   图片2（蓝、紫、白与透明珠钻，大小不一、形状各异），图片3（粉色点钻笔与透明小盒中的
   粉色点钻胶）。构图自然美观，视觉主体居中且饱满突出，无左右留白限制（不限制安全区）。
   全程为正常真实播放速度，无慢动作。全程无任何文字或字幕。视频结束时 bgm 也刚好结束。」

── 逐镜头段（每个镜头一段，段与段之间用换行分开）──

  格式：「镜头 N（起始秒-结束秒） 景别与运镜 + 焦距(mm) + 光圈(f/…)。场景 + 主体位置与动作过程（带量化：角度/幅度/速度/停留时间）+ 产品呈现与性能 + 光位光型/材质反光。画面文字。」
  —— 镜头**不加小标题**；每个镜头写足细节、约 **90-160 字**，要**详细到可复刻原镜头**。必须做到以下四点量化：
     ① 运镜量化：给出**景别 + 焦距**（如 35mm 广角 / 85mm 中焦 / 100mm 微距）**+ 光圈**（如 f/2.0、f/2.8、f/4）；
        运镜写清**静止/推/拉/摇/移** 及**速度与时长**（如"缓慢右移，约每秒 2cm，持续约 4 秒"；静止则写"机位静止、画面稳定"）。
     ② 动作量化：主体动作写出**角度、幅度、速度、停留/持续时间**（如"指尖与画面呈约 45 度角，轻触后停留约 1 秒再移开"）。
     ③ 主体细节：必要的**手型 / 指甲（如'指甲修剪整齐'）/ 持握方式 / 接触点**等可见细节。
     ④ 光影材质：**光位光型**（如"侧光""3200K 暖光自左 45 度透入""顶部冷色射灯"）+ **材质反光**（如"钻石在侧光下反射微光，凸起立体质感清晰可见"）。
  ⚠️ **画面文字/字幕忠实原广告**：原广告有花字/字幕/slogan 时，写出其**具体英文文案**与出现方式（位置、动效，如"中央急速浮现/淡入/滑入"）；
     原广告无文字才写"无字幕"。**不凭空加、不删原有字幕**。

  示例（学 ✓ 的量化详略；细到可复刻，非堆形容词）：
  ✓ 微距触碰类：「镜头 2（2.43-5.53秒） 静止特写，焦距 100mm，光圈 f/2.8，机位静止、画面稳定。聚焦于钻石贴片画（与图片1一致）
     的整体构图与细节；画面右侧，一只女性的手（指甲修剪整齐）从画面外缓慢伸入，指尖轻触画作右侧边缘一颗蓝色圆形钻石贴片，
     指尖与画面呈约 45 度角，轻触后停留约 1 秒再迅速移开。钻石在侧光下反射微光，凸起的立体质感与可触碰感清晰可见。无字幕。」
  ✓ 运镜类：「镜头 3（5.53-7.56秒） 俯视中景，焦距 35mm，光圈 f/4，机位缓慢下压约每秒 2cm、持续约 2 秒，背景轻微虚化。
     画面右上方一只女性右手持一小袋白色钻石，缓慢倾倒白钻至白色托盘，随后依次倾倒蓝钻、紫钻，堆叠出斑斓色块（与图片2一致），
     珠钻在柔光下细碎反光。无字幕。」
  ✓ 有花字类：「镜头 1（0.0-2.0秒） 中景跟拍，焦距 50mm，光圈 f/2.0，快节奏横移跟随。暗调硬核健身房，顶部冷色射灯；
     肌肉饱满、带真实汗水的美国男模双手甩动沉重战绳，动作狂暴富爆发力，上衣（与图片1一致）以 3D 剪裁贴合肌肉轮廓、无多余褶皱，
     面料表面在硬光下呈哑光质感。画面正中由中央急速浮现无衬线英文花字 "ENGINEERED FOR POWER"。」

── 硬性要求 ──

- 全局约束段包含上述要素，参考图内容只在此段写一次
- 逐镜头段每段必含：镜头 N + 时段 + 景别 + **焦距(mm) + 光圈(f/…)** + 运镜（含速度/时长）+ 场景 + 主体动作（含**角度/幅度/停留时间**）+ 光位光型/材质反光 + 画面文字（花字文案或"无字幕"）
- 焦距/光圈/速度/角度/停留时间等**数值可合理估计**（用"约"字），但必须给出，使镜头可被复刻
- 画面文字忠实原广告：有则写出具体英文文案 + 出现方式，无则"无字幕"，不凭空增删
- 时段严格按 shot 顺序从 0 累积
- 删除：镜头小标题；"展示…以突出…""强调…""营造…"等目的/解说词；氛围情绪评价；对参考图内容的重复复述（改"与图片X一致"）
- **产品性能 / 材质 / 动作细节要保留**；冗余定语精简（如"女性的右手食指指腹"→"指腹"）
- 每个镜头约 90-160 字：量化到可复刻生成，又不靠形容词堆砌凑长

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

    prompt_text = _prompt('video_prompt', PROMPT).format(
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
                print(f"    [VideoPrompt] usage: prompt={usage.prompt_tokens} / completion={usage.completion_tokens} / cost=${_usage_cost(usage):.6f}")
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
