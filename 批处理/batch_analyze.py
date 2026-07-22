"""
Phase 2: 批处理入口
===================

输入：根目录（下每个子文件夹是一条广告的镜头集）
流程：
  1. 扫子文件夹（含 shot_*.mp4 的算一条广告）
  2. 对每个广告：调 合并/拆分流程 生成 shot_*.json + ad_aggregate.json
                → 调 video_prompt_generator 生成 video_prompt.json
  3. 全部完成后，汇总所有 video_prompt.json → <root>/横向对比.md

用法：
  python 批处理/batch_analyze.py <root> --product "..." [选项]

选项：
  --product "..."               产品名（必填，全部广告共用）
  --pipeline merged|split       默认 merged
  --pro                         用 Gemini 2.5 Pro
  --overwrite                   强制重跑（否则只补跑缺失的产物）
  --no-cross-compare            跳过横向对比 md 生成
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
import subprocess
import argparse
import traceback
import time
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# 让本文件可以导入同目录下的 video_prompt_generator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_prompt_generator import (
    generate_video_prompt,
    _make_client,
    _repair_llm_json,
)
from shot_prompt_generator import generate_shot_prompts
from common_shot_prompt_generator import generate_common_shot_prompts


# ============================================================
# ⚠️ 配置区（密钥从环境变量 / .env 读取，不再硬编码）
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MERGED_RUN = PROJECT_ROOT / "合并流程" / "run.py"
SPLIT_RUN = PROJECT_ROOT / "拆分流程" / "run.py"

# ============================================================
# 横向对比 Schema
# ============================================================

class 广告对比行(BaseModel):
    ad_id: str
    时长: str = Field(description="如 '15.2s'")
    主色_SKU: str = Field(description="配色 / SKU 数量描述")
    文案结构: str = Field(description="主结构 + 副结构，如 '演示型+直给'")
    钩子类型: str = Field(description="如 '结果前置+悬念口播'、'揭袋反差'、'体量反差'")
    前3秒画面: str = Field(description="简述前 3 秒画面")
    节奏: str = Field(description="如 '~5段 / 2.5s · 中快'")
    口播: str = Field(description="有 / 无 / 清晰 / 被音乐盖")
    卖点字幕: str = Field(description="有 / 无 / 密集 / 贯穿")
    音乐: str = Field(description="背景乐 / 同款治愈歌 等")
    创意角度: str = Field(description="'人群 × 卖点 × 情绪' 格式")
    值得复用点: str = Field(description="一句话点睛")


class 横向对比(BaseModel):
    品类: str = Field(description="一句话概括，如 '户外人造仿真花'")
    样本数: int
    对比行: List[广告对比行]
    共性套路: List[str] = Field(min_length=1, description="3-5 条，每条一段话")
    最小配方: str = Field(description="100-150 字一段，给出复用套路的最小执行公式")


COMPARE_PROMPT = """你是资深广告策略分析师。基于以下 N 条广告的视频级 prompt 数据，
生成一份横向对比分析报告（结构参照传统广告投放策略报告）。

【N 条广告数据】
{ads_json}

【任务】

1. 品类：一句话概括这批广告的品类
2. 横向对比表：为每条广告填写 11 个维度
   - 时长
   - 主色/SKU：配色、SKU 数量描述
   - 文案结构：主结构 + 副结构（PAS / 直给型 / Before-After / UGC证言体 / 演示型）
   - 钩子类型：如 "结果前置+悬念口播"、"揭袋反差"、"体量反差"、"开箱悬念"
   - 前3秒画面：简述
   - 节奏：格式如 "~5段 / 2.5s · 中快"
   - 口播：有 / 无 / 清晰 / 被音乐盖
   - 卖点字幕：有 / 无 / 密集 / 贯穿
   - 音乐：背景乐 / 同款治愈歌 等
   - 创意角度：格式 "人群 × 卖点 × 情绪"，如 "懒打理屋主 × 免维护/显贵/一包搞定 × 省心惊喜"
   - 值得复用点：一句话点睛

3. 共性套路：3-5 条，每条一段话，深入分析共通做法（结构 / 钩子 / 卖点话术 / 音乐 / 素材来源）
4. 最小配方：一段（100-150 字），给出复用套路的最小可执行公式

【风格】
参考老派广告策略洞察报告：直接、有洞察、有节奏感、可复用。
注意："运镜"维度整体可靠性低，如果表达就明确标"待人工确认"。

严格按下面 JSON schema 输出，不要 markdown 代码块：

{schema_json}
"""


# ============================================================
# 工具函数
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


def find_ad_folders(root: Path) -> List[Path]:
    """扫描根目录下含 shot_*.mp4 的子文件夹。"""
    ad_folders = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if list(sub.glob("shot_*.mp4")):
            ad_folders.append(sub)
    return ad_folders


def get_analysis_dir(ad_folder: Path, pipeline: str) -> Path:
    subdir = "analysis_output" if pipeline == "merged" else "analysis_output_split"
    return ad_folder / subdir


def check_analysis_done(ad_folder: Path, pipeline: str) -> bool:
    return (get_analysis_dir(ad_folder, pipeline) / "ad_aggregate.json").exists()


def run_pipeline_for_ad(
    ad_folder: Path, product: str, pipeline: str, pro: bool, overwrite: bool,
    hypotheses: Optional[str] = None
) -> Optional[Path]:
    """
    跑单条广告的分析流程。
    - overwrite=True：无论如何都跑（传 --overwrite 给底层）
    - overwrite=False 且已有产物：跳过
    - overwrite=False 且没产物：跑（补跑失败的）
    """
    run_script = MERGED_RUN if pipeline == "merged" else SPLIT_RUN
    if not run_script.exists():
        raise RuntimeError(f"找不到 {run_script}")

    if check_analysis_done(ad_folder, pipeline) and not overwrite:
        print(f"    [跳过分析] {get_analysis_dir(ad_folder, pipeline).name} 已存在")
        return get_analysis_dir(ad_folder, pipeline)

    cmd = [
        sys.executable, str(run_script),
        str(ad_folder),
        "--product", product,
    ]
    if pro:
        cmd.append("--pro")
    if overwrite:
        cmd.append("--overwrite")
    if hypotheses:
        cmd += ["--hypotheses", hypotheses]

    print(f"    [跑 {pipeline}] {' '.join(cmd[:2])} ... --product {product}")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print(f"    [失败] 底层流程返回码 {proc.returncode}")
        return None

    if not check_analysis_done(ad_folder, pipeline):
        return None
    return get_analysis_dir(ad_folder, pipeline)


def process_ad(
    ad_folder: Path, product: str, pipeline: str, pro: bool, overwrite: bool,
    gen_shot_prompts: bool = True, shot_prompts_min_count: int = 7,
    hypotheses: Optional[str] = None,
) -> dict:
    """
    处理一条广告：分析 → video_prompt.json → shot_prompts.json。
    返回 {"ad_id", "status", "video_prompt": Path or None, "shot_prompts": Path or None, "error"}
    """
    ad_id = ad_folder.name
    result = {
        "ad_id": ad_id, "status": "unknown",
        "video_prompt": None, "shot_prompts": None,
        "error": None,
    }
    model = "google/gemini-2.5-pro" if pro else "google/gemini-2.5-flash"

    try:
        # Step A：分析
        analysis_dir = run_pipeline_for_ad(ad_folder, product, pipeline, pro, overwrite, hypotheses)
        if analysis_dir is None:
            result["status"] = "分析失败"
            return result

        # Step B：video_prompt.json
        vp_path = analysis_dir / "video_prompt.json"
        if vp_path.exists() and not overwrite:
            print(f"    [跳过 VideoPrompt] 已存在 {vp_path.name}")
        else:
            generate_video_prompt(
                analysis_dir=analysis_dir,
                product_name=product,
                output_path=vp_path,
                model=model,
                api_key=OPENROUTER_API_KEY,
            )
        result["video_prompt"] = vp_path

        # Step C：shot_prompts.json（可选）
        if gen_shot_prompts:
            sp_path = analysis_dir / "shot_prompts.json"
            if sp_path.exists() and not overwrite:
                print(f"    [跳过 ShotPrompts] 已存在 {sp_path.name}")
            else:
                generate_shot_prompts(
                    video_prompt_path=vp_path,
                    output_path=sp_path,
                    min_count=shot_prompts_min_count,
                    model=model,
                    api_key=OPENROUTER_API_KEY,
                )
            result["shot_prompts"] = sp_path

        result["status"] = "成功"
    except Exception as e:
        # video_prompt 都出了错就当整体失败；只 shot_prompts 出错则标"部分成功"
        if result["video_prompt"] is None:
            result["status"] = "VideoPrompt 生成失败"
        else:
            result["status"] = "ShotPrompts 生成失败（VideoPrompt 已成功）"
        result["error"] = str(e)
        traceback.print_exc()

    return result


# ============================================================
# 横向对比 md 生成
# ============================================================

def _call_compare_llm(prompt_text: str, model: str) -> str:
    client = _make_client(OPENROUTER_API_KEY)
    is_pro = "pro" in model.lower()
    req_kwargs = (
        {"max_tokens": 16000, "extra_body": {"reasoning": {"effort": "low", "exclude": True}}}
        if is_pro else
        {"max_tokens": 12000, "extra_body": {"reasoning": {"max_tokens": 0, "exclude": True}}}
    )

    print(f"[横向对比] 用 {model} 生成汇总...")

    last_err = None
    for attempt in range(1, 4):
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
                print(f"[横向对比] usage: prompt={usage.prompt_tokens} / completion={usage.completion_tokens} / cost=${_usage_cost(usage):.6f}")
            finish = resp.choices[0].finish_reason
            raw = (resp.choices[0].message.content or "").strip()
            if finish == "error" or (usage and usage.completion_tokens == 0) or len(raw) < 30:
                if attempt < 3:
                    wait = 5 * attempt
                    print(f"[横向对比] [重试 {attempt}/3] finish={finish}, 等 {wait}s")
                    time.sleep(wait)
                    continue
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].lstrip()
            return raw
        except Exception as e:
            last_err = e
            if attempt < 3:
                wait = 5 * attempt
                print(f"[横向对比] [重试 {attempt}/3] 异常: {e}，等 {wait}s")
                time.sleep(wait)
            continue

    raise RuntimeError(f"横向对比 LLM 调用失败: {last_err}")


def generate_cross_compare(vp_results: List[dict], root: Path, model: str) -> Optional[Path]:
    """
    汇总所有成功的 video_prompt.json，调 LLM 生成横向对比 md。
    """
    ok_vps = [r for r in vp_results if r["video_prompt"] is not None]
    failed = [r for r in vp_results if r["video_prompt"] is None]

    if not ok_vps:
        print("[横向对比] 无成功广告，跳过")
        return None

    ads_data = []
    for r in ok_vps:
        with open(r["video_prompt"], "r", encoding="utf-8") as f:
            data = json.load(f)
        ads_data.append({
            "ad_id": data.get("ad_id"),
            "total_duration": data.get("total_duration"),
            "n_shots": data.get("n_shots"),
            "产品名": data.get("产品名"),
            "广告结构分类": data.get("广告结构分类"),
            "video_prompt_text": data.get("video_prompt_text", ""),
            "整体流转逻辑": data.get("整体流转逻辑"),
            "商品卖点": data.get("商品卖点"),
            "解决痛点": data.get("解决痛点"),
            "镜头列表": [
                {
                    "序号": s.get("序号"),
                    "时段": s.get("时段"),
                    "标题": s.get("标题"),
                    "主要卖点": s.get("主要卖点"),
                    "配合卖点": s.get("配合卖点"),
                }
                for s in data.get("镜头列表", [])
            ],
        })

    prompt_text = _prompt('compare', COMPARE_PROMPT).format(
        ads_json=json.dumps(ads_data, ensure_ascii=False, indent=2),
        schema_json=json.dumps(横向对比.model_json_schema(), ensure_ascii=False, indent=2),
    )

    raw = _call_compare_llm(prompt_text, model)
    raw = _repair_llm_json(raw)

    try:
        compare = 横向对比.model_validate_json(raw)
    except Exception as e1:
        try:
            import json_repair  # type: ignore
            fixed = json_repair.loads(raw)
            compare = 横向对比.model_validate(fixed)
        except Exception:
            debug_path = root / "debug_cross_compare_raw.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"--- 首次错误 ---\n{e1}\n\n--- RAW ---\n{raw}")
            raise RuntimeError(f"横向对比 JSON 解析失败，原始响应存到 {debug_path}")

    md = _assemble_compare_md(compare, failed)
    out_path = root / "横向对比.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[横向对比] 已保存 → {out_path}")
    return out_path


def _assemble_compare_md(compare: 横向对比, failed: List[dict]) -> str:
    """把结构化数据组装成 markdown。"""
    ad_ids = [r.ad_id for r in compare.对比行]

    lines = []
    lines.append(f"# {compare.品类} · 短视频广告横向对比汇总\n")
    lines.append(f"> 样本：{compare.样本数} 条素材（{'、'.join(ad_ids)}）")
    lines.append(f"> 分析方式：合并/拆分流程 → video_prompt.json → LLM 横向汇总")
    lines.append(f"> ⚠️ 部分维度（尤其运镜、素材来源）可靠性待人工确认")
    lines.append("")
    lines.append("## 一、横向对比表\n")

    header = "| 维度 | " + " | ".join(ad_ids) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(ad_ids)) + " |"
    lines.append(header)
    lines.append(sep)

    dims = [
        ("时长", "时长"),
        ("主色/SKU", "主色_SKU"),
        ("文案结构", "文案结构"),
        ("钩子类型", "钩子类型"),
        ("前3秒画面", "前3秒画面"),
        ("节奏(镜头段/均长)", "节奏"),
        ("口播", "口播"),
        ("卖点字幕", "卖点字幕"),
        ("音乐", "音乐"),
        ("创意角度(人群×卖点×情绪)", "创意角度"),
        ("值得复用点", "值得复用点"),
    ]
    for dim_label, attr in dims:
        row = f"| **{dim_label}** | " + " | ".join(
            getattr(r, attr, "-") or "-" for r in compare.对比行
        ) + " |"
        lines.append(row)

    lines.append("")
    lines.append("## 二、这批素材的共性套路（一眼看穿）\n")
    for i, item in enumerate(compare.共性套路, 1):
        lines.append(f"**{i}.** {item}\n")

    lines.append("")
    lines.append("## 三、复用这批套路的最小配方\n")
    lines.append(f"> {compare.最小配方}")

    if failed:
        lines.append("")
        lines.append("---")
        lines.append("## 附：分析失败的广告\n")
        for f in failed:
            err_str = f" ({f['error']})" if f.get("error") else ""
            lines.append(f"- **{f['ad_id']}**：{f['status']}{err_str}")

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="批处理广告分析 + 横向对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("root", help="根目录（下有多个广告子文件夹）")
    p.add_argument("--product", required=True, help="产品名（所有广告共用）")
    p.add_argument("--pipeline", choices=["merged", "split"], default="merged",
                   help="用哪条分析流程（默认 merged）")
    p.add_argument("--pro", action="store_true", help="用 gemini-2.5-pro")
    p.add_argument("--overwrite", action="store_true", help="强制重跑（不加则只补跑缺失的产物）")
    p.add_argument("--cross-compare", action="store_true", help="生成横向对比 md（默认关闭）")
    p.add_argument("--hypotheses", default=None, help="前置卖点提示（可选，逗号分隔），供分析模型参考")
    p.add_argument("--no-shot-prompts", action="store_true", help="跳过每条广告的 shot_prompts.json 生成")
    p.add_argument("--shot-prompts-count", type=int, default=7, help="每条广告生成的独立镜头 prompt 数量（默认 7）")
    p.add_argument("--no-common-shot-prompts", action="store_true", help="跳过品类级共性镜头 prompt 生成")
    p.add_argument("--common-shot-prompts-count", type=int, default=7, help="品类级共性镜头 prompt 数量（默认 7）")
    return p.parse_args()


def main():
    args = parse_args()

    if not OPENROUTER_API_KEY or not OPENROUTER_API_KEY.startswith("sk-or-"):
        print("[错误] 请在 batch_analyze.py 顶部填入有效的 OPENROUTER_API_KEY")
        sys.exit(1)
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

    root = Path(args.root)
    if not root.is_dir():
        print(f"[错误] 根目录不存在: {root}")
        sys.exit(1)

    ad_folders = find_ad_folders(root)
    if not ad_folders:
        print(f"[错误] 未在 {root} 找到含 shot_*.mp4 的子文件夹")
        sys.exit(1)

    print(f"\n发现 {len(ad_folders)} 条广告：")
    for f in ad_folders:
        print(f"  - {f.name}")
    print(f"\n产品名: {args.product}")
    print(f"流程:   {args.pipeline}")
    print(f"模型:   {'gemini-2.5-pro' if args.pro else 'gemini-2.5-flash'}")
    print(f"覆盖:   {'是（强制重跑）' if args.overwrite else '否（补跑缺失）'}")
    print(f"单广告镜头 prompt 素材库: {'跳过' if args.no_shot_prompts else f'生成 (≥{args.shot_prompts_count} 条)'}")
    print(f"品类共性镜头 prompt:     {'跳过' if args.no_common_shot_prompts else f'生成 (≥{args.common_shot_prompts_count} 条)'}")
    print(f"横向对比 md:             {'生成' if args.cross_compare else '跳过（默认，加 --cross-compare 开启）'}")
    print()

    results = []
    for i, ad_folder in enumerate(ad_folders, 1):
        print(f"\n=== [{i}/{len(ad_folders)}] {ad_folder.name} ===")
        r = process_ad(
            ad_folder, args.product, args.pipeline, args.pro, args.overwrite,
            gen_shot_prompts=not args.no_shot_prompts,
            shot_prompts_min_count=args.shot_prompts_count,
            hypotheses=args.hypotheses,
        )
        results.append(r)
        print(f"    → 状态: {r['status']}")

    # 统计
    print("\n" + "=" * 50)
    print("汇总")
    print("=" * 50)
    ok = sum(1 for r in results if r["status"] == "成功")
    print(f"成功: {ok} / 总计: {len(results)}")
    for r in results:
        if r["status"] != "成功":
            err = f" ({r['error']})" if r.get("error") else ""
            print(f"  ✗ {r['ad_id']}: {r['status']}{err}")

    model = "google/gemini-2.5-pro" if args.pro else "google/gemini-2.5-flash"

    # Step D：品类级共性镜头 prompt（跨广告）
    if not args.no_common_shot_prompts:
        try:
            common_path = root / "共性镜头prompt.json"
            if common_path.exists() and not args.overwrite:
                print(f"\n[共性镜头prompt] 跳过（已存在 {common_path.name}，加 --overwrite 覆盖）")
            else:
                print()
                generate_common_shot_prompts(
                    root=root,
                    product_name=args.product,
                    output_path=common_path,
                    min_count=args.common_shot_prompts_count,
                    pipeline=args.pipeline,
                    model=model,
                    api_key=OPENROUTER_API_KEY,
                )
        except Exception as e:
            print(f"[共性镜头prompt] 失败: {e}")
            traceback.print_exc()

    # Step E：横向对比 md（默认关闭，加 --cross-compare 开启）
    if args.cross_compare:
        try:
            print()
            generate_cross_compare(results, root, model=model)
        except Exception as e:
            print(f"[横向对比] 失败: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
