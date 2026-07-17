"""
镜头级动作三层分解 + 卖点绑定（方案 B：CoT 三轮推理）
======================================================

Coarse（人工给）→ Fine（Round 1 拆）→ Atomic（Round 2 拆）→ 卖点绑定（Round 3）

Round 1: Fine 分解     —— 把 coarse 动作拆成 2-5 个叙事级子动作
Round 2: Atomic 分解   —— 每个 Fine 再拆成 0.3-1.5 秒的原子动作 + 产品/运镜/屏幕
Round 3: 卖点绑定      —— 找协同事件 + 从每个原子动作推卖点 + 证据链

最终输出的 ShotAnalysis 与"合并流程"完全同 schema，便于对比。

依赖：
    pip install openai pydantic
环境变量：
    OPENROUTER_API_KEY=sk-or-v1-...
"""

from __future__ import annotations
import os
import base64
import json
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ============================================================
# 一、共用 Schema（与合并流程完全一致，保证可对比）
# ============================================================

class HumanAction(BaseModel):
    id: str = Field(description="唯一 id，如 A1, A2")
    ts: List[float] = Field(description="[开始秒, 结束秒]", min_length=2, max_length=2)
    actor: str = Field(description="谁：female / male / KOL / user / 无人")
    body_part: str = Field(description="身体部位：食指指腹 / 掌心 / 双手 / 面部...")
    verb: str = Field(description="动词：取 / 搓 / 推 / 点 / 按 / 拍 / 扫 / 涂 / 抹")
    direction: Optional[str] = Field(default=None, description="方向")
    target: str = Field(description="作用对象")
    force: Optional[str] = Field(default=None, description="力度/节奏")
    note: Optional[str] = None


class ProductPresentation(BaseModel):
    id: str = Field(description="唯一 id，如 B1, B2")
    ts: List[float] = Field(min_length=2, max_length=2)
    type: Literal[
        "静态展示", "旋转展示", "倾倒", "挤出", "开合",
        "拆解", "特写放大", "参数飘出", "对比展示", "使用演示",
        "剂型呈现", "内部结构展示"
    ]
    focus: str = Field(description="强调点：包装 / 材质 / 剂型 / 颜色 / 内部结构 / 参数 / 认证")
    technique: Optional[str] = Field(default=None, description="拍摄手法：微距 / 慢镜头 / 定格")
    note: Optional[str] = None


class CameraMovement(BaseModel):
    ts: List[float] = Field(min_length=2, max_length=2)
    motion: str = Field(description="运动 + 景别，如 '静止中景' / '推近至面部特写' / '环绕微距'")


class ScreenOverlay(BaseModel):
    ts: List[float] = Field(min_length=2, max_length=2)
    text: str
    element_type: Literal["标题", "副标题", "参数", "CTA", "认证", "价格", "品牌", "图标"] = "标题"
    style: Optional[str] = None


class ActionChains(BaseModel):
    human: List[HumanAction] = Field(default_factory=list)
    product: List[ProductPresentation] = Field(default_factory=list)
    camera: List[CameraMovement] = Field(default_factory=list)
    overlay: List[ScreenOverlay] = Field(default_factory=list)


class SellingPoint(BaseModel):
    id: str
    name: str
    feature: str
    benefit: str
    evidence: List[str] = Field(min_length=1)
    modality: Literal["视觉动作", "视觉展示", "口播", "屏幕文字", "多模态协同"] = "视觉展示"
    confidence: float = Field(ge=0.0, le=1.0)


class ShotAnalysis(BaseModel):
    shot_id: str
    start_ts: float
    end_ts: float
    shot_intent_one_liner: str
    action_chains: ActionChains
    coactive_events: List[str] = Field(default_factory=list)
    selling_points: List[SellingPoint] = Field(default_factory=list)


# ============================================================
# 二、中间层 Schema（Round 1 / Round 2）
# ============================================================

class FineAction(BaseModel):
    """Round 1 输出的 Fine 级子动作。"""
    id: str = Field(description="F1, F2, ...")
    ts: List[float] = Field(min_length=2, max_length=2)
    description: str = Field(description="叙事级动作描述，如 '取用面霜'、'预热掌心'、'分区涂抹'")
    body_parts: List[str] = Field(default_factory=list)
    product_state: str = Field(description="该阶段产品状态，如 '瓶中'、'挤出手心'、'涂到脸上'")


class Round1Output(BaseModel):
    shot_id: str
    shot_intent_one_liner: str = Field(description="一句话概括镜头意图")
    fine_actions: List[FineAction]


class Round2Output(BaseModel):
    shot_id: str
    action_chains: ActionChains = Field(description="完整四条链，human 里每个 HumanAction 的 note 需带上 parent_fine_id，如 'from F2'")


# ============================================================
# 三、Prompt 模板（三轮）
# ============================================================

ROUND1_PROMPT = """你是资深广告镜头拆解分析师。这是【第 1 轮】：Fine 级动作分解。

【人工提示】
{hint_block}

【镜头信息】
- shot_id: {shot_id}
- 时长: {duration:.2f} 秒

【任务】把这个镜头的动作拆成 2-5 个 Fine 级子动作。

Fine 是"叙事级动作"：比 coarse（如"涂抹面霜"）细，比 atomic（如"食指指腹取用"）粗，
每个 Fine 代表一个完整的叙事步骤，例如：
  - "取用" / "预热" / "分区涂抹" / "轻拍收尾"
  - "揭开外盒" / "取出商品" / "呈现最终形态"

对每个 Fine 输出：
- id: F1, F2, ...
- ts: [开始秒, 结束秒]（相对镜头开始）
- description: 叙事级动作描述
- body_parts: 涉及的身体部位（可空）
- product_state: 该阶段产品状态

【硬约束】
- 时间戳来自你观察到的实际视觉切换点，**严禁均匀等分**
- 无法确定精确时间时宁可少拆几段
- 若镜头是纯静态展示或无人物动作，fine_actions 可返回极少条目或按产品状态划分

严格 JSON 输出，不要 markdown 代码块：

{schema_json}
"""


ROUND2_PROMPT = """你是资深广告镜头拆解分析师。这是【第 2 轮】：Atomic 原子动作分解。

【人工提示】
{hint_block}

【镜头信息】
- shot_id: {shot_id}
- 时长: {duration:.2f} 秒

【第 1 轮 Fine 分解结果】
{fine_json}

【任务】对每个 Fine 子动作，进一步拆成 Atomic 原子动作（每条 0.3-1.5 秒），
同时输出完整的产品展示链、运镜链、屏幕元素链。

Atomic 原子动作要求：
- 每个 Atomic 归属某个 Fine，在 note 开头明确写 "from F1" / "from F2" 等
- 每个 Atomic 含：actor / body_part / verb / direction / target / force / note
- verb 用单动词：取 / 搓 / 推 / 点 / 按 / 拍 / 扫 / 涂 / 抹 / 展开 / 放下 / 握住 / 拉扯 / 撕开 / 定型 / 抖动
- body_part 具体到部位（食指指腹 / 掌心 / 左手 / 双手 / 上半身 / 膝盖）
- direction 必填：如 "由内向外"、"由上至下"、"画圆"、"上下"、"左右"
- force 必填：轻 / 中 / 重 / 快 / 慢 / 匀速 / 停顿
- note **必填 30-80 字分镜脚本级描述**，含：起始位置 / 身体姿态 / 与产品交互 / 与相机相对位置 / 结束状态。
  ✅ "from F2. 女性从画面左侧门框内走两步到纸箱前，膝盖微弯，上身前倾约30度，
      双手同时伸进纸箱两侧，握住绣球花外包装两侧凹槽处"
  ❌ "from F2. 双手拆包装"（太笼统）

产品展示链（B1, B2, ...）：
- 每条描述产品自身的展示阶段（静态展示 / 倾倒 / 挤出 / 拆解 / 特写放大 / 剂型呈现 等）
- focus 说明强调点（包装 / 材质 / 剂型 / 内部结构 / 参数 等）
- technique 若有微距/慢镜头/定格等手法也填上
- note **必填 20-50 字**：产品当下状态 + 视觉焦点 + 与人物交互关系
  ✅ "绣球花置于纸箱内，透明塑料膜完整包覆花瓣，透过塑料可见浅蓝紫渐变色"

运镜链（camera）：
- motion 必填 20-50 字，含运镜路径 + 景别变化 + 与被摄对象的角度关系
  ✅ "俯视 30 度中景 → 缓慢推近至花朵特写，最终花头占画面 70%"
  ❌ "静止中景"

屏幕元素链（overlay）：
- 若无屏幕文字/图标出现，返回空数组
- 有的话要具体到内容 + 出现位置

【硬约束】
- 时间戳来自实际观察，严禁均分
- direction / force / note 必填
- Atomic 时间戳落在其父 Fine 的时段内
- 多个 Atomic 的 note 串起来要能拼成**完整的分镜脚本**（读起来像剧本）

严格 JSON 输出：

{schema_json}
"""


ROUND3_PROMPT = """你是资深广告镜头卖点绑定分析师。这是【第 3 轮】：卖点归纳 + 证据链。

【人工提示】
{hint_block}

【镜头信息】
- shot_id: {shot_id}
- 时长: {duration:.2f} 秒

【前两轮已生成】
Fine 分解：{fine_json}

四条动作链（Atomic + 产品 + 运镜 + 屏幕）：
{chains_json}

【任务】
1. 先找出跨链"协同事件"（Atomic + 产品展示 + 运镜 + 屏幕文字里，同一时段发生的组合）；
   协同事件通常对应本镜头卖点强调点。

2. 从每个 Atomic 动作和协同事件推卖点：
   - 该动作/组合在广告里传达什么产品价值？
   - 本镜头是否真实展示了这个价值？给出证据（引用 A1/B2/camera/overlay 或 OCR/ASR 原文）
   - 该价值解决什么用户痛点？（把痛点写在 benefit 里）

3. 合并同类卖点，避免重复。

【硬约束（继承）】

  1. 卖点必须是"产品本身的价值"，不能是"广告演示手法"。
     反例（不要输出）：
       - "包装好拆" —— 那是演示，不是产品价值
       - "画面清晰" —— 那是拍摄技术
     正例：
       - "免养护"、"送礼场景合适"、"替代真花的痛点"

  2. 尝试从下列角度**深度归纳**卖点：
     - 场景使用（何时用、在哪用）
     - 情感价值（安心感/仪式感/惊喜感/身份认同）
     - 替代方案对比（相比 X 更好）
     - 目标人群的隐性痛点

  3. 每个卖点：
     - name：一句话
     - feature：客观特征
     - benefit：用户利益（含所解决的痛点）
     - evidence：**必须**引用具体链 id
     - modality：证据模态
     - confidence：0–1

  4. 无证据的卖点不要输出；协同事件催生的卖点优先；纯过渡镜头 selling_points 可返回 []。

【最终输出】完整的 ShotAnalysis JSON。把 Round 1/2 的动作链原样填入 action_chains，
补充 coactive_events 和 selling_points。start_ts=0，end_ts={duration}，
shot_intent_one_liner 沿用 Round 1 的（可微调）。

严格按下面 JSON schema 输出，不要输出多余文字、不要 markdown 代码块：

{schema_json}
"""


# ============================================================
# 四、调用 OpenRouter
# ============================================================

def build_hint_block(
    product: str,
    action: Optional[str] = None,
    hypotheses: Optional[List[str]] = None,
    audience: Optional[str] = None,
) -> str:
    lines = [f"- 产品: {product}"]
    if action:
        lines.append(f"- 动作/操作: {action}")
    if hypotheses:
        lines.append(f"- 待验证假设（可确认/否定/补充）: {', '.join(hypotheses)}")
    if audience:
        lines.append(f"- 目标受众提示: {audience}")
    return "\n".join(lines)


def _make_client(api_key: Optional[str] = None):
    from openai import OpenAI
    import httpx
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("缺少 OPENROUTER_API_KEY。")
    http_client = httpx.Client(timeout=300.0)
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        http_client=http_client,
        default_headers={
            "HTTP-Referer": "https://local.shot-analyzer",
            "X-Title": "Shot Analyzer (CoT 3-round)",
        },
    )


def _build_content(prompt_text: str, video_bytes: Optional[bytes],
                   keyframes: Optional[List[bytes]], shot_id: str) -> list:
    content = [{"type": "text", "text": prompt_text}]
    if video_bytes:
        b64 = base64.b64encode(video_bytes).decode("ascii")
        content.append({
            "type": "file",
            "file": {
                "filename": f"{shot_id}.mp4",
                "file_data": f"data:video/mp4;base64,{b64}",
            },
        })
    else:
        for kf in (keyframes or []):
            b64 = base64.b64encode(kf).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
    return content


def _call_openrouter(client, model: str, content: list, round_label: str,
                     max_tokens: int, is_pro: bool, max_retries: int = 5) -> str:
    import time
    if is_pro:
        req_kwargs = {
            "max_tokens": max_tokens,
            "extra_body": {"reasoning": {"effort": "medium", "exclude": True}},
        }
        thinking_desc = "开启 (effort=medium)"
    else:
        req_kwargs = {
            "max_tokens": max_tokens,
            "extra_body": {"reasoning": {"max_tokens": 0, "exclude": True}},
        }
        thinking_desc = "关闭"

    print(
        f"    [{round_label}] model={model}, thinking={thinking_desc}, max_tokens={max_tokens}"
    )

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0.1,
                **req_kwargs,
            )
            usage = getattr(resp, "usage", None)
            if usage:
                print(
                    f"    [{round_label}] usage: prompt={usage.prompt_tokens} / "
                    f"completion={usage.completion_tokens} / total={usage.total_tokens}"
                )
            raw = (resp.choices[0].message.content or "").strip()
            finish = resp.choices[0].finish_reason

            # 检测无效响应
            token_zero = usage and usage.completion_tokens == 0
            hard_fail = (finish == "error" and token_zero) or len(raw) < 30

            if hard_fail and attempt < max_retries:
                # 硬失败（上游拒了）：更长的退避
                wait = min(5 * attempt, 30)
                print(f"    [{round_label}] [重试 {attempt}/{max_retries}] finish={finish}, len={len(raw)}, 等 {wait}s")
                time.sleep(wait)
                continue

            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].lstrip()

            if finish != "stop":
                print(f"    [{round_label}] [警告] finish_reason={finish}，可能截断（长度 {len(raw)}）")
            return raw
        except Exception as e:
            last_err = e
            wait = min(5 * attempt, 30)
            print(f"    [{round_label}] [重试 {attempt}/{max_retries}] 异常: {e}，等 {wait}s")
            if attempt < max_retries:
                time.sleep(wait)
            continue

    raise RuntimeError(f"[{round_label}] 重试 {max_retries} 次仍失败：{last_err}")


def analyze_shot(
    shot_id: str,
    duration: float,
    keyframes: Optional[List[bytes]] = None,
    video_bytes: Optional[bytes] = None,
    video_mime: str = "video/mp4",
    asr_segments: Optional[List[dict]] = None,   # 保留接口一致，本流程不用
    ocr_regions: Optional[List[dict]] = None,
    detection_results: Optional[List[dict]] = None,
    human_hint: Optional[dict] = None,
    model: str = "google/gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> ShotAnalysis:
    """
    三轮 CoT 分析：
      Round 1: Fine 分解
      Round 2: Atomic + 完整四条链
      Round 3: 协同事件 + 卖点归纳（最终 ShotAnalysis）
    """
    if not video_bytes and not keyframes:
        raise ValueError("必须提供 video_bytes 或 keyframes 之一")

    client = _make_client(api_key)
    is_pro = "pro" in model.lower()
    hint_block = build_hint_block(**(human_hint or {"product": "未知"}))

    # ---------- Round 1: Fine 分解 ----------
    r1_prompt = ROUND1_PROMPT.format(
        hint_block=hint_block,
        shot_id=shot_id,
        duration=duration,
        schema_json=json.dumps(Round1Output.model_json_schema(), ensure_ascii=False, indent=2),
    )
    r1_content = _build_content(r1_prompt, video_bytes, keyframes, shot_id)
    r1_raw = _call_openrouter(
        client, model, r1_content, "R1-Fine",
        max_tokens=8000 if not is_pro else 16000, is_pro=is_pro
    )
    r1_raw = _repair_llm_json(r1_raw)
    r1 = Round1Output.model_validate_json(r1_raw)
    print(f"    [R1-Fine] {len(r1.fine_actions)} 个 Fine 动作")

    # 轮次间小间隔，减轻 OpenRouter 上游压力
    import time as _time
    _time.sleep(1.0)

    # ---------- Round 2: Atomic + 四条链 ----------
    r2_prompt = ROUND2_PROMPT.format(
        hint_block=hint_block,
        shot_id=shot_id,
        duration=duration,
        fine_json=r1.model_dump_json(indent=2),
        schema_json=json.dumps(Round2Output.model_json_schema(), ensure_ascii=False, indent=2),
    )
    r2_content = _build_content(r2_prompt, video_bytes, keyframes, shot_id)
    r2_raw = _call_openrouter(
        client, model, r2_content, "R2-Atomic",
        max_tokens=12000 if not is_pro else 24000, is_pro=is_pro
    )
    r2_raw = _repair_llm_json(r2_raw)
    r2 = Round2Output.model_validate_json(r2_raw)
    print(
        f"    [R2-Atomic] human={len(r2.action_chains.human)} / "
        f"product={len(r2.action_chains.product)} / "
        f"camera={len(r2.action_chains.camera)} / "
        f"overlay={len(r2.action_chains.overlay)}"
    )

    _time.sleep(1.0)

    # ---------- Round 3: 协同 + 卖点归纳 ----------
    r3_prompt = ROUND3_PROMPT.format(
        hint_block=hint_block,
        shot_id=shot_id,
        duration=duration,
        fine_json=r1.model_dump_json(indent=2),
        chains_json=r2.model_dump_json(indent=2),
        schema_json=json.dumps(ShotAnalysis.model_json_schema(), ensure_ascii=False, indent=2),
    )
    r3_content = _build_content(r3_prompt, video_bytes, keyframes, shot_id)
    r3_raw = _call_openrouter(
        client, model, r3_content, "R3-SellingPoints",
        max_tokens=16000 if not is_pro else 32000, is_pro=is_pro
    )
    r3_raw = _repair_llm_json(r3_raw)

    try:
        result = ShotAnalysis.model_validate_json(r3_raw)
    except Exception as e1:
        try:
            import json_repair  # type: ignore
            fixed = json_repair.loads(r3_raw)
            result = ShotAnalysis.model_validate(fixed)
        except Exception:
            debug_path = f"debug_{shot_id}_r3_raw.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"--- 首次错误 ---\n{e1}\n\n--- RAW ---\n{r3_raw}")
            raise RuntimeError(f"R3 JSON 解析失败，原始响应存 {debug_path}")

    # 兜底：把 shot_id / start_ts / end_ts / 四条链填对
    result.shot_id = shot_id
    result.start_ts = 0.0
    result.end_ts = round(duration, 3)
    # R2 的动作链一般优于 R3 里模型的重述，硬替换保证一致性
    if r2.action_chains.human or r2.action_chains.product:
        result.action_chains = r2.action_chains

    return result


# ============================================================
# 五、JSON 修复
# ============================================================

def _repair_llm_json(raw: str) -> str:
    import re
    raw = re.sub(r'(,\s*)[a-zA-Z_][a-zA-Z_0-9]*"\s*:\s*', r'\1', raw)
    raw = re.sub(r'^\s*[a-zA-Z_][a-zA-Z_0-9]*"\s*:\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r':\s*"null"\s*(,|\n|\})', r': null\1', raw)
    raw = re.sub(r',(\s*[\]\}])', r'\1', raw)
    return raw


# ============================================================
# 六、证据校验（同合并流程）
# ============================================================

def validate_evidence(
    result: ShotAnalysis,
    asr_segments: Optional[List[dict]] = None,
    ocr_regions: Optional[List[dict]] = None,
) -> ShotAnalysis:
    all_ids = set()
    for a in result.action_chains.human:
        all_ids.add(a.id)
    for b in result.action_chains.product:
        all_ids.add(b.id)
    all_ids |= {"camera", "overlay"}

    asr_full = " ".join(s["text"] for s in (asr_segments or []))
    ocr_full = " ".join(r["text"] for r in (ocr_regions or []))

    for sp in result.selling_points:
        ok = 0
        for ev in sp.evidence:
            hit = any(cid in ev for cid in all_ids)
            if not hit and asr_full:
                hit = any(w in asr_full for w in ev.split() if len(w) >= 2)
            if not hit and ocr_full:
                hit = any(w in ocr_full for w in ev.split() if len(w) >= 2)
            if hit:
                ok += 1
        total = len(sp.evidence)
        if total > 0:
            ratio = ok / total
            if ratio < 1.0:
                sp.confidence *= (0.5 + 0.5 * ratio)

    return result


# ============================================================
# 七、商业化整合 Schema + 汇总（与合并流程完全一致）
# ============================================================

class 展示方式集合(BaseModel):
    人物动作: List[str] = Field(default_factory=list, description="分镜脚本级详细描述，30-80字/条。含起始位置+身体姿态+身体部位+与产品交互+与相机相对位置。示例：'女性从画面左侧门框内走两步到纸箱前，膝盖微弯，上身前倾约30度，双手同时伸进纸箱两侧握住绣球花外包装，缓慢向上提至齐额头高度，正面朝向相机'")
    产品展示: List[str] = Field(default_factory=list, description="20-50字/条，含产品状态+视觉焦点+拍摄手法")
    运镜: List[str] = Field(default_factory=list, description="20-50字/条，含运镜路径+景别变化")
    屏幕元素: List[str] = Field(default_factory=list, description="5-20字/条，具体到内容+出现位置")


class 分镜摘要(BaseModel):
    shot_id: str
    时段: str = Field(description="如 '0.00-2.96s'")
    镜头意图: str = Field(description="10-30 字，说清楚该镜头在广告故事线里的角色")
    人物动作: List[str] = Field(default_factory=list, description="分镜脚本级详细描述，30-80字/条")
    产品展示: List[str] = Field(default_factory=list, description="20-50字/条")
    运镜方式: List[str] = Field(default_factory=list, description="含运镜路径+景别变化，20-50字/条")
    主要卖点: str = Field(description="1 条最核心卖点短标签，4-8字")
    配合卖点: List[str] = Field(min_length=2, description="至少 2 条短标签，2-5字")


class 广告整合(BaseModel):
    ad_id: str
    total_duration: float
    n_shots: int
    商品卖点: List[str]
    解决痛点: List[str]
    展示方式: 展示方式集合
    广告主张: str
    分镜摘要: List[分镜摘要]


AGGREGATE_PROMPT = """你是广告分析质检交付编辑。以下是一条广告的完整分镜级分析。
请汇总成商业化质检交付格式。

【核心原则：卖点用短标签，动作/展示/运镜要极详细，像分镜脚本】

═══ 广告级字段 ═══

  1. 商品卖点：每条 3-8 字**短标签**，全广告去重
     ✅ "高仿真度"、"开箱即用"、"免养护打理"、"高档礼盒感"
     ❌ 长句子

  2. 解决痛点：每条 3-10 字**短标签**
     ✅ "养不活鲜花"、"送礼焦虑"

  3. 展示方式（**必须详细**）：
     - 人物动作：**30-80 字/条**，分镜脚本级
       ✅ "女性从画面左侧门框内走两步到纸箱前，膝盖微弯，上身前倾约30度，
           双手同时伸进纸箱两侧握住绣球花外包装，缓慢向上提至齐额头高度"
       ❌ "双手拆箱"（太笼统）
     - 产品展示：**20-50 字/条**，含产品状态 + 视觉焦点 + 拍摄手法
     - 运镜：**20-50 字/条**，含运镜路径 + 景别变化
       ❌ "静止中景"（无信息量，不列）
     - 屏幕元素：5-20 字/条，具体到内容 + 出现位置

  4. 广告主张：一句不超过 30 字

═══ 每个镜头（分镜摘要）═══

  - 时段：如 "0.00-2.96s"
  - 镜头意图：10-30 字，说清楚该镜头在故事线里的角色
  - 主要卖点：**1 个短标签**，4-8 字
  - 配合卖点：**至少 2 个短标签**，2-5 字
  - 人物动作：**30-80 字/条**，可 1-3 条，分镜脚本级
  - 产品展示：**20-50 字/条**
  - 运镜方式：**20-50 字/条**，含运镜路径 + 景别变化

═══ 关键要求 ═══
  - 多个镜头的人物动作串起来要形成**完整的产品展示故事线**
  - 语义相近的卖点合并
  - 无信息量的通用运镜不列

镜头数据（时间戳已是整段广告全局时间）：

{shots_json}

严格按下面 JSON schema 输出：

{schema_json}
"""


def summarize_to_business_json(
    shots: List[dict],
    ad_id: str,
    total_duration: float,
    model: str = "google/gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> 广告整合:
    client = _make_client(api_key)

    slim = []
    for s in shots:
        chains = s.get("action_chains", {}) or {}
        slim.append({
            "shot_id": s.get("shot_id"),
            "时段": f'{s.get("start_ts", 0):.2f}-{s.get("end_ts", 0):.2f}s',
            "镜头意图": s.get("shot_intent_one_liner", ""),
            "human_actions_full": [
                {
                    "verb": a.get("verb", ""),
                    "body_part": a.get("body_part", ""),
                    "direction": a.get("direction"),
                    "target": a.get("target", ""),
                    "force": a.get("force"),
                    "note": a.get("note", ""),
                }
                for a in chains.get("human", [])
            ],
            "product_full": [
                {
                    "type": b.get("type", ""),
                    "focus": b.get("focus", ""),
                    "technique": b.get("technique"),
                    "note": b.get("note", ""),
                }
                for b in chains.get("product", [])
            ],
            "camera_motions": [c.get("motion", "") for c in chains.get("camera", [])],
            "overlay": [f'"{o.get("text","")}"' for o in chains.get("overlay", [])],
            "coactive_events": s.get("coactive_events", []),
            "selling_points": [
                {"name": sp.get("name"), "feature": sp.get("feature"), "benefit": sp.get("benefit")}
                for sp in s.get("selling_points", [])
            ],
        })

    prompt_text = AGGREGATE_PROMPT.format(
        shots_json=json.dumps(slim, ensure_ascii=False, indent=2),
        schema_json=json.dumps(广告整合.model_json_schema(), ensure_ascii=False, indent=2),
    )

    is_pro = "pro" in model.lower()
    req_kwargs = (
        {"max_tokens": 16000, "extra_body": {"reasoning": {"effort": "low", "exclude": True}}}
        if is_pro else
        {"max_tokens": 12000, "extra_body": {"reasoning": {"max_tokens": 0, "exclude": True}}}
    )

    print(f"[汇总] 用 {model} 生成商业化整合 JSON...")

    import time
    last_err = None
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                response_format={"type": "json_object"},
                temperature=0.2,
                **req_kwargs,
            )
            raw = (resp.choices[0].message.content or "").strip()
            finish = resp.choices[0].finish_reason
            usage = getattr(resp, "usage", None)
            if usage:
                print(f"[汇总] usage: prompt={usage.prompt_tokens} / completion={usage.completion_tokens}")

            if finish == "error" or (usage and usage.completion_tokens == 0):
                print(f"[汇总] [重试 {attempt}/3] finish={finish}")
                time.sleep(2 * attempt)
                continue

            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].lstrip()

            if finish != "stop":
                print(f"[汇总] [警告] finish_reason={finish}，可能截断（长度 {len(raw)}）")

            raw = _repair_llm_json(raw)

            # 三级修复：直接解析 → 内置修复 → json_repair
            try:
                result = 广告整合.model_validate_json(raw)
            except Exception as e1:
                try:
                    import json_repair  # type: ignore
                    fixed = json_repair.loads(raw)
                    result = 广告整合.model_validate(fixed)
                    print(f"[汇总] json_repair 修复成功")
                except Exception:
                    raise e1

            result.ad_id = ad_id
            result.total_duration = round(total_duration, 3)
            result.n_shots = len(shots)
            return result

        except Exception as e:
            last_err = e
            print(f"[汇总] [重试 {attempt}/3] 异常: {e}")
            if attempt < 3:
                time.sleep(2 * attempt)
            continue

    raise RuntimeError(f"汇总重试 3 次仍失败：{last_err}")
