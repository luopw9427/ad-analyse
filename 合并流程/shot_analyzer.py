"""
镜头级动作链拆解 + 卖点归纳
==========================

基于 OpenRouter（Gemini 2.5 Flash）+ Pydantic 强制结构化输出。

主线任务：
    输入一个镜头 → 输出四条时间线（人物/产品/运镜/屏幕）+ 协同事件 + 卖点归纳。

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
# 一、Pydantic Schema —— 强制结构化输出
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
    evidence: List[str] = Field(
        min_length=1,
        description="证据：必须引用具体链 id（A1/B2/camera/overlay）或 OCR/ASR 原文"
    )
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
# 二、Prompt 模板
# ============================================================

PROMPT = """你是资深广告镜头拆解分析师。请把这一个镜头拆解为「四条时间线 + 协同事件 + 卖点归纳」。

【人工提示】
{hint_block}

【镜头基本信息】
- shot_id: {shot_id}
- 时长: {duration:.2f} 秒

【关键帧】按时序附在下方（第 1 张 = 镜头起始，最后一张 = 镜头结束）

【ASR 口播（带时间戳，秒）】
{asr_text}

【屏幕文字 OCR】
{ocr_text}

【物体检测（开放词表 Grounding）】
{detection_text}

---

【任务 A：拆四条时间线】原子级（每条事件 0.3–1.5 秒），时间戳相对本镜头开始（0 → {duration:.2f}）：

  1. human —— 人物动作原子事件
     必须像**分镜脚本**一样详细，note 字段是关键。示例（这才叫"够详细"）：
       ✅ "女性从画面左侧门框内走两步到纸箱前，膝盖微弯，上身向前倾约30度，
           双手同时伸进纸箱两侧，握住绣球花外包装，缓慢向上提起至齐额头高度，
           花朵与相机形成正面视角"
       ✅ "右手食指指腹按压花瓣中心，手指与花瓣呈45度角，轻压 0.3 秒后抬起，
           展示花瓣柔软的回弹感"
       ❌ "双手悬挂花盆" —— 太笼统，缺细节
       ❌ "拆开包装" —— 完全没有身体位置/角度/路径信息

     每条 human 事件必填：
       - id / ts / actor
       - body_part：具体部位（食指指腹 / 掌心 / 左手 / 双手 / 手肘 / 上半身 / 整个身体）
       - verb：单动作动词
       - direction：方向（由内向外 / 从左至右 / 由下至上 / 面向相机...）
       - target：明确对象（不是"包装"，而是"绣球花外层塑料膜"）
       - force：力度/节奏（轻/中/重 + 快/慢/停顿）
       - note：**必填 30-80 字的分镜脚本描述**，必须包含：
         · 起始位置（画面里在哪、从哪来）
         · 身体姿态（弯腰几度、手抬到哪、身体朝向）
         · 与产品的交互方式（怎么接触、接触在产品的哪个部位）
         · 与相机的相对位置（正面 / 侧面 / 背对）
         · 动作节奏与结束状态

  2. product —— 产品展示原子事件
     必须写清楚：产品**呈现什么状态**、**镜头聚焦到产品哪个细节**、**用什么手法拍**。
     note 字段 20-50 字。例：
       ✅ "绣球花置于纸箱内，花头朝上，透明塑料膜完整包覆花瓣，
           透过塑料可见花瓣颜色为浅蓝紫渐变"
       ✅ "移除包装后花朵瞬间蓬松展开，直径由原 15cm 恢复到 25cm，
           花瓣层次分明呈球形团簇"
       ❌ "静态展示花朵" —— 没描述产品哪个特征被强调

  3. camera —— 相机运动 + 景别（motion 字段 20-50 字，示例）：
       ✅ "俯视 30 度中景 → 缓慢推近至花朵特写，最终框住花头占画面 70%"
       ✅ "手持晃动 → 稳定，从人物腰部平摇至面部特写"
       ❌ "静止中景" —— 无信息量

  4. overlay —— 屏幕文字与图标（标题 / 副标题 / 参数 / CTA / 认证 / 价格 / 品牌 / 图标）

【任务 B：协同事件】找出四条链里"同一时段"发生的原子事件组合。
描述涉及的链 id 组合 + 组合含义，这是本镜头的卖点强调点。

【任务 C：卖点归纳】从动作链和协同事件推出本镜头卖点。

  每个卖点必须：
    - name：一句话名字
    - feature：客观特征
    - benefit：用户利益
    - evidence：**必须**引用具体链 id（A1/B2/camera/overlay）或 OCR/ASR 原文
    - modality：证据模态
    - confidence：0–1

  硬规则：
    - 无证据的卖点不要输出
    - 优先输出由协同事件催生的卖点
    - 若本镜头是纯过渡镜头，selling_points 返回 []
    - 不要重复卖点

【硬约束（违反视为无效输出）】

  1. 时间戳必须来自你观察到的实际视觉切换点，**严禁把镜头时长均分成等长片段**
     （比如 4 秒镜头输出 [0,1][1,2][2,3][3,4] 这种整齐切分几乎必错）。
     无法确定精确时间时，宁可少拆几段。

  2. HumanAction 的 direction / force **必填**，note **必写 30-80 字分镜脚本描述**。
     note 必须体现：起始位置 → 姿态 → 与产品交互 → 与相机相对位置 → 结束状态。
     product 的 note 必须描述产品当下状态 + 视觉焦点。
     camera 的 motion 必须描述运镜路径 + 景别变化。

     多个镜头的 human 动作串起来应该像分镜脚本，能让观众看到**完整的产品使用/展示故事**：
     从"发现产品"→"接近产品"→"接触产品"→"操作产品"→"展示效果"→"结束姿态"这条链。
     所以每条 human 事件的 note 里要明确它在故事线里承担的角色。

  3. 卖点必须是"产品本身的价值"，不能是"广告演示手法"。
     反例（不要输出）：
       - "包装好拆" —— 那是演示，不是产品价值
       - "画面清晰" —— 那是拍摄技术
     正例：
       - "免养护"、"送礼场景合适"、"替代真花的痛点"

  4. 尝试从下列角度**深度归纳**卖点（不局限于表面动作）：
       - 场景使用（何时用、在哪用）
       - 情感价值（安心感/仪式感/惊喜感/身份认同）
       - 替代方案对比（相比 X 更好）
       - 目标人群的隐性痛点（懒/时间紧/审美追求/送礼焦虑）

严格按下面 JSON schema 输出，不要输出多余文字、不要 markdown 代码块：

{schema_json}
"""


# ============================================================
# 三、调用 OpenRouter
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
        raise RuntimeError(
            "缺少 OPENROUTER_API_KEY。请设置环境变量或调用时传 api_key=。"
        )
    # 自建 httpx.Client 绕过 openai 老版本与 httpx 0.28 的 proxies 参数不兼容 bug
    http_client = httpx.Client(timeout=180.0)
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        http_client=http_client,
        default_headers={
            "HTTP-Referer": "https://local.shot-analyzer",
            "X-Title": "Shot Analyzer",
        },
    )


def analyze_shot(
    shot_id: str,
    duration: float,
    keyframes: Optional[List[bytes]] = None,
    video_bytes: Optional[bytes] = None,
    video_mime: str = "video/mp4",
    asr_segments: Optional[List[dict]] = None,
    ocr_regions: Optional[List[dict]] = None,
    detection_results: Optional[List[dict]] = None,
    human_hint: Optional[dict] = None,
    model: str = "google/gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> ShotAnalysis:
    """
    分析单个镜头。二选一：
      - video_bytes: 整段 mp4 字节（Gemini 原生视频输入，推荐）
      - keyframes:  关键帧 JPEG 字节列表（降级方案）
    """
    if not video_bytes and not keyframes:
        raise ValueError("必须提供 video_bytes 或 keyframes 之一")
    client = _make_client(api_key)

    hint_block = build_hint_block(**(human_hint or {"product": "未知"}))

    asr_text = "\n".join(
        f"  [{s['ts'][0]:.1f}s - {s['ts'][1]:.1f}s] {s['text']}"
        for s in (asr_segments or [])
    ) or "  (无)"

    ocr_text = "\n".join(
        f"  frame {r['frame']}: \"{r['text']}\" @ bbox={r['bbox']}"
        for r in (ocr_regions or [])
    ) or "  (无)"

    det_text = "\n".join(
        f"  frame {d['frame']}: {d['label']} bbox={d['bbox']} conf={d['conf']:.2f}"
        for d in (detection_results or [])
    ) or "  (无)"

    prompt_text = PROMPT.format(
        hint_block=hint_block,
        shot_id=shot_id,
        duration=duration,
        asr_text=asr_text,
        ocr_text=ocr_text,
        detection_text=det_text,
        schema_json=json.dumps(
            ShotAnalysis.model_json_schema(), ensure_ascii=False, indent=2
        ),
    )

    # OpenAI-compatible messages 格式
    content = [{"type": "text", "text": prompt_text}]

    if video_bytes:
        # Gemini 原生视频：走 OpenRouter 的 file 类型
        b64 = base64.b64encode(video_bytes).decode("ascii")
        content.append({
            "type": "file",
            "file": {
                "filename": f"{shot_id}.mp4",
                "file_data": f"data:{video_mime};base64,{b64}",
            },
        })
    else:
        # 关键帧降级方案
        for kf_bytes in keyframes:
            b64 = base64.b64encode(kf_bytes).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

    # 按模型区分：Pro 强制开 reasoning、需要更大 max_tokens；Flash 关掉 reasoning
    is_pro = "pro" in model.lower()
    if is_pro:
        # Pro 不允许关 reasoning，给足 max_tokens 让正文写得完
        req_kwargs = {
            "max_tokens": 32000,
            "extra_body": {"reasoning": {"effort": "medium", "exclude": True}},
        }
        thinking_desc = "开启 (effort=medium, 不返回思考内容)"
    else:
        # Flash 关掉 thinking，避免消耗输出预算导致 JSON 截断
        req_kwargs = {
            "max_tokens": 16000,
            "extra_body": {"reasoning": {"max_tokens": 0, "exclude": True}},
        }
        thinking_desc = "关闭"

    # 打印本次调用的实际参数
    video_kb = len(video_bytes) / 1024 if video_bytes else 0
    n_kf = len(keyframes) if keyframes else 0
    input_mode = f"原生视频 {video_kb:.0f}KB" if video_bytes else f"关键帧 {n_kf} 张"
    prompt_chars = len(prompt_text)
    print(
        f"    ├─ model={model}\n"
        f"    ├─ input={input_mode}, prompt={prompt_chars} 字符\n"
        f"    ├─ thinking={thinking_desc}\n"
        f"    ├─ max_tokens={req_kwargs['max_tokens']}, temperature=0.1\n"
        f"    └─ response_format=json_object"
    )

    # 自动重试：应对 OpenRouter 上游 error / 0 token
    import time as _time
    max_retries = 5
    resp = None
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
                    f"    ├─ usage: prompt={usage.prompt_tokens} / "
                    f"completion={usage.completion_tokens} / total={usage.total_tokens}"
                )
            finish = resp.choices[0].finish_reason
            raw_check = (resp.choices[0].message.content or "").strip()
            token_zero = usage and usage.completion_tokens == 0
            hard_fail = (finish == "error" and token_zero) or len(raw_check) < 30
            if hard_fail and attempt < max_retries:
                wait = min(5 * attempt, 30)
                print(f"    [重试 {attempt}/{max_retries}] finish={finish}, len={len(raw_check)}, 等 {wait}s")
                _time.sleep(wait)
                continue
            break
        except Exception as e:
            last_err = e
            wait = min(5 * attempt, 30)
            print(f"    [重试 {attempt}/{max_retries}] 异常: {e}，等 {wait}s")
            if attempt < max_retries:
                _time.sleep(wait)
            continue
    if resp is None:
        raise RuntimeError(f"重试 {max_retries} 次仍失败：{last_err}")

    raw = (resp.choices[0].message.content or "").strip()

    # 保底剥掉可能的 markdown 代码块
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    # 若响应被截断，把 finish_reason 打出来方便定位
    finish = resp.choices[0].finish_reason
    if finish != "stop":
        print(f"  [警告] finish_reason={finish}，响应可能被截断（长度 {len(raw)} 字符）")

    try:
        return ShotAnalysis.model_validate_json(raw)
    except Exception as e1:
        # 尝试自动修复 LLM 生成的常见 JSON 错误
        repaired = _repair_llm_json(raw)
        try:
            result = ShotAnalysis.model_validate_json(repaired)
            print(f"    [修复] JSON 自动修复成功")
            return result
        except Exception:
            pass
        # 尝试使用 json_repair 库（如已安装）
        try:
            import json_repair  # type: ignore
            fixed_obj = json_repair.loads(raw)
            result = ShotAnalysis.model_validate(fixed_obj)
            print(f"    [修复] 使用 json_repair 修复成功")
            return result
        except ImportError:
            pass
        except Exception:
            pass
        # 保存原始响应便于排查
        debug_path = f"debug_{shot_id}_raw.txt"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(
                f"finish_reason: {finish}\n\n"
                f"--- 首次错误 ---\n{e1}\n\n"
                f"--- RAW ---\n{raw}\n\n"
                f"--- REPAIRED (仍失败) ---\n{repaired}"
            )
        raise RuntimeError(f"JSON 解析失败，原始响应已存到 {debug_path}")


def _repair_llm_json(raw: str) -> str:
    """轻量修复 LLM 生成 JSON 的常见错误。"""
    import re
    # 1. 修复数组/对象中意外的裸露字段名前缀
    #    如：`0.0,\nend_ts": 2.0` -> `0.0,\n2.0`
    raw = re.sub(r'(,\s*)[a-zA-Z_][a-zA-Z_0-9]*"\s*:\s*', r'\1', raw)
    # 2. 修复以裸字段名开头的行（缺少前导引号）
    raw = re.sub(r'^\s*[a-zA-Z_][a-zA-Z_0-9]*"\s*:\s*', '', raw, flags=re.MULTILINE)
    # 3. 字符串 "null" -> JSON null
    raw = re.sub(r':\s*"null"\s*(,|\n|\})', r': null\1', raw)
    # 4. 去掉数组/对象末尾多余的逗号
    raw = re.sub(r',(\s*[\]\}])', r'\1', raw)
    return raw


# ============================================================
# 3.5 商业化整合 Schema + 汇总
# ============================================================

class 展示方式集合(BaseModel):
    人物动作: List[str] = Field(default_factory=list, description="分镜脚本级详细描述，含位置/姿态/身体部位/与产品交互/与相机相对位置，30-80字/条，如 '女性从画面左侧门框内走两步到纸箱前，膝盖微弯，上身前倾约30度，双手同时伸进纸箱两侧握住绣球花外包装，缓慢向上提起至齐额头高度'")
    产品展示: List[str] = Field(default_factory=list, description="详细描述，含产品状态+视觉焦点+拍摄手法，20-50字/条")
    运镜: List[str] = Field(default_factory=list, description="详细运镜路径 + 景别变化，20-50字/条，如 '俯视 30 度中景缓慢推近至花朵特写，最终花头占画面 70%'")
    屏幕元素: List[str] = Field(default_factory=list, description="具体到内容 + 出现位置，5-20字/条")


class 分镜摘要(BaseModel):
    shot_id: str
    时段: str = Field(description="如 '0.00-2.96s'")
    镜头意图: str = Field(description="10-30 字，说清楚这镜头在故事线里的角色")
    人物动作: List[str] = Field(default_factory=list, description="分镜脚本级详细描述，30-80字/条。含起始位置+身体姿态+身体部位+与产品交互+与相机相对位置。示例：'女性从画面左侧门框内走两步到纸箱前，膝盖微弯，上身前倾约30度，双手同时伸进纸箱两侧握住绣球花外包装，缓慢向上提至齐额头高度，正面朝向相机'")
    产品展示: List[str] = Field(default_factory=list, description="含产品状态+视觉焦点+拍摄手法，20-50字/条。示例：'绣球花置于纸箱内，透明塑料膜完整包覆花瓣，透过塑料可见花瓣呈浅蓝紫渐变'")
    运镜方式: List[str] = Field(default_factory=list, description="含运镜路径+景别变化，20-50字/条。示例：'俯视 30 度中景 → 缓慢推近至花朵特写，最终花头占画面 70%'")
    主要卖点: str = Field(description="1 条最核心卖点短标签，4-8字，如 '高仿真度'")
    配合卖点: List[str] = Field(min_length=2, description="至少 2 条配合卖点短标签，2-5字，如 ['免养护', '开箱即用', '高档奢侈']")


class 广告整合(BaseModel):
    ad_id: str
    total_duration: float
    n_shots: int
    商品卖点: List[str] = Field(description="每条 2-5 字短标签，如 '好打理'、'开箱即用'、'仿真度高'")
    解决痛点: List[str] = Field(description="每条 2-8 字，如 '养不活鲜花'、'打理麻烦'")
    展示方式: 展示方式集合
    广告主张: str = Field(description="一句话不超过 25 字")
    分镜摘要: List[分镜摘要]


AGGREGATE_PROMPT = """你是广告分析质检交付编辑。以下是一条广告的完整分镜级分析。
请汇总成商业化质检交付格式。

【核心原则：卖点用短标签，动作/展示/运镜要极详细】

═══ 广告级字段 ═══

  1. 商品卖点：每条 3-8 字**短标签**，全广告去重
     ✅ "高仿真度"、"开箱即用"、"免养护打理"、"高档礼盒感"、"送礼首选"
     ❌ 长句子；❌ 相近的分别列（合并成一条）

  2. 解决痛点：每条 3-10 字**短标签**
     ✅ "养不活鲜花"、"打理麻烦"、"送礼焦虑"、"凋谢太快"

  3. 展示方式（**必须详细**，四类分别汇总）：
     - 人物动作：**30-80 字/条**，分镜脚本级详细描述
       ✅ "女性从画面左侧门框内走两步到纸箱前，膝盖微弯，上身前倾约30度，
           双手同时伸进纸箱两侧握住绣球花外包装，缓慢向上提至齐额头高度，
           正面朝向相机"
       ❌ "双手拆箱"、"取出花朵展示"（太笼统，不合格）
     - 产品展示：**20-50 字/条**，含产品状态 + 视觉焦点 + 拍摄手法
       ✅ "绣球花置于纸箱内，透明塑料膜完整包覆花瓣，透过塑料可见浅蓝紫渐变"
       ❌ "静态展示"、"包装呈现"（太笼统）
     - 运镜：**20-50 字/条**，含运镜路径 + 景别变化
       ✅ "俯视 30 度中景 → 缓慢推近至花朵特写，最终花头占画面 70%"
       ❌ "静止中景"（无信息量，不列）
     - 屏幕元素：5-20 字/条，具体到内容 + 出现位置

  4. 广告主张：一句不超过 30 字

═══ 每个镜头（分镜摘要）═══

  必填字段：
  - 时段：如 "0.00-2.96s"
  - 镜头意图：10-30 字，说清楚该镜头在整个广告故事线里承担的角色
  - 主要卖点：**1 个**短标签，4-8 字
    ✅ "高仿真度"
  - 配合卖点：**至少 2 个**短标签，2-5 字
    ✅ ["免养护", "开箱即用", "高档奢侈"]
  - 人物动作：**30-80 字/条**，可 1-3 条，分镜脚本级详细描述
     必须描述：起始位置 → 身体姿态与部位 → 与产品交互方式 → 与相机相对位置
  - 产品展示：**20-50 字/条**，产品状态 + 视觉焦点 + 拍摄手法
  - 运镜方式：**20-50 字/条**，含运镜路径 + 景别变化

═══ 关键要求 ═══

  - 所有镜头的人物动作串起来要形成**完整的产品展示故事线**：
    从"发现产品"→"接近产品"→"接触产品"→"操作产品"→"展示效果"→"结束状态"
  - 每个镜头的人物动作 note 必须体现它在这条故事线里的位置
  - 语义相近的卖点合并为一条最凝练的表述
  - 无信息量的通用运镜（如"普通中景"）不列

═══ 面向人工质检 ═══
  输出要让读者扫完就能：
  - 卖什么、解决什么（看卖点/痛点短标签）
  - 怎么演的（看动作/展示/运镜的分镜脚本级描述）
  - 每个镜头承担什么故事角色

镜头数据（时间戳已是整段广告全局时间）：

{shots_json}

严格按下面 JSON schema 输出，不要输出多余文字、不要 markdown 代码块：

{schema_json}
"""


def summarize_to_business_json(
    shots: List[dict],
    ad_id: str,
    total_duration: float,
    model: str = "google/gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> 广告整合:
    """把所有 shot 分析结果送 Gemini 汇总成商业化交付 JSON。"""
    client = _make_client(api_key)

    # 精简 shot 数据，只送对汇总有用的字段
    slim = []
    for s in shots:
        chains = s.get("action_chains", {}) or {}
        slim.append({
            "shot_id": s.get("shot_id"),
            "时段": f'{s.get("start_ts", 0):.2f}-{s.get("end_ts", 0):.2f}s',
            "镜头意图": s.get("shot_intent_one_liner", ""),
            # 保留完整 note（分镜脚本描述）供汇总层生成详细的人物动作字段
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
            "overlay": [
                f'"{o.get("text","")}"' for o in chains.get("overlay", [])
            ],
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
        {"max_tokens": 8000, "extra_body": {"reasoning": {"effort": "low", "exclude": True}}}
        if is_pro else
        {"max_tokens": 6000, "extra_body": {"reasoning": {"max_tokens": 0, "exclude": True}}}
    )

    print(f"[汇总] 用 {model} 生成商业化整合 JSON...")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        response_format={"type": "json_object"},
        temperature=0.2,
        **req_kwargs,
    )

    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    result = 广告整合.model_validate_json(raw)
    # 硬填入正确的元信息（LLM 可能算错时长/镜头数）
    result.ad_id = ad_id
    result.total_duration = round(total_duration, 3)
    result.n_shots = len(shots)
    return result


# ============================================================
# 四、证据校验（抑制幻觉）
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
        total = len(sp.evidence)
        for ev in sp.evidence:
            hit = any(cid in ev for cid in all_ids)
            if not hit and asr_full:
                hit = any(w in asr_full for w in ev.split() if len(w) >= 2)
            if not hit and ocr_full:
                hit = any(w in ocr_full for w in ev.split() if len(w) >= 2)
            if hit:
                ok += 1
        if total > 0:
            ratio = ok / total
            if ratio < 1.0:
                sp.confidence *= (0.5 + 0.5 * ratio)

    return result
