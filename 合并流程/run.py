"""
批处理入口：选文件夹 → 顺序读取 shot_XXX.mp4 → 抽帧 → 调 Gemini → 保存 JSON

用法：
    python run.py                          # 弹出文件夹选择框
    python run.py D:/path/to/shots         # 直接传路径
    python run.py D:/path/to/shots --product "无油配方面霜" --action "涂抹" --audience "敏感肌"

依赖：
    pip install openai pydantic opencv-python
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import List, Tuple

import cv2  # opencv-python

from shot_analyzer import analyze_shot, validate_evidence, summarize_to_business_json


# ============================================================
# ⚠️ 配置区（换 key / 换默认路径时改这里）
# ============================================================
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# 默认镜头文件夹（不传参数时用这个；命令行传路径时会覆盖它）
DEFAULT_FOLDER = r"F:\实习\云熙汇智\视频拆分\拆分示例"

# 只处理这个通配符匹配的文件（自动忽略文件夹里其他杂文件）
DEFAULT_PATTERN = "shot_*.mp4"

# 是否走 Gemini 原生视频输入（推荐 True：更省 token、能捕捉动作时序）
# 若 mp4 大小 > MAX_VIDEO_MB 自动降级为抽帧
USE_NATIVE_VIDEO = True
MAX_VIDEO_MB = 20
# ============================================================


# ============================================================
# 抽帧 + 时长
# ============================================================

def get_video_duration(video_path: Path) -> float:
    """只读时长，不抽帧。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps if fps > 0 else 0.0


def extract_keyframes(video_path: Path, n_frames: int = 6, max_side: int = 720) -> Tuple[List[bytes], float]:
    """
    从 mp4 均匀抽取 n_frames 张关键帧，返回 JPEG 字节列表 + 时长（秒）。
    自动降采样到长边 <= max_side 以省 token。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = total / fps if fps > 0 else 0.0

    if total <= 0:
        cap.release()
        return [], 0.0

    if total <= n_frames:
        indices = list(range(total))
    else:
        # 均匀取 n_frames 张（含首帧和末帧）
        indices = [int(round(i * (total - 1) / (n_frames - 1))) for i in range(n_frames)]

    frames_bytes: List[bytes] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            frames_bytes.append(buf.tobytes())

    cap.release()
    return frames_bytes, duration


# ============================================================
# 文件夹选择
# ============================================================

def pick_folder_dialog() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title="选择包含 shot_XXX.mp4 的文件夹")
        root.destroy()
        return path or ""
    except Exception:
        return ""


# ============================================================
# 主流程
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="批量分析 shot_*.mp4 → 输出 JSON")
    p.add_argument("folder", nargs="?", help="包含 shot_XXX.mp4 的文件夹（不填走 DEFAULT_FOLDER，为空则弹出选择框）")
    p.add_argument("--product", default="未知产品", help="人工提示：产品名")
    p.add_argument("--action", default=None, help="人工提示：动作（可选）")
    p.add_argument("--audience", default=None, help="人工提示：目标受众（可选）")
    p.add_argument("--hypotheses", default=None, help="人工提示：待验证卖点，逗号分隔（可选）")
    p.add_argument("--model", default="google/gemini-2.5-flash", help="OpenRouter 模型名")
    p.add_argument("--n-frames", type=int, default=6, help="每镜头抽帧数（默认 6）")
    p.add_argument("--overwrite", action="store_true", help="覆盖已有结果")
    p.add_argument("--pattern", default=DEFAULT_PATTERN, help=f"文件名通配符（默认 {DEFAULT_PATTERN}）")
    p.add_argument("--frames", action="store_true", help="强制走关键帧模式（默认走 Gemini 原生视频）")
    p.add_argument("--pro", action="store_true", help="用 gemini-2.5-pro（更强，贵一档）")
    p.add_argument("--no-aggregate", action="store_true", help="不生成整合级 ad_aggregate.json")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. 校验 API key
    if not OPENROUTER_API_KEY or not OPENROUTER_API_KEY.startswith("sk-or-"):
        print("[错误] 请在 run.py 顶部填入有效的 OPENROUTER_API_KEY")
        sys.exit(1)
    # 注入到环境变量，供 shot_analyzer 读取
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

    # 2. 确定文件夹：命令行参数 > DEFAULT_FOLDER > 弹窗
    folder = args.folder or DEFAULT_FOLDER or pick_folder_dialog()
    if not folder:
        print("[错误] 未选择文件夹")
        sys.exit(1)

    folder = Path(folder)
    if not folder.is_dir():
        print(f"[错误] 不是目录: {folder}")
        sys.exit(1)

    # 3. 扫文件
    shots = sorted(folder.glob(args.pattern))
    if not shots:
        print(f"[错误] 在 {folder} 未找到 {args.pattern} 文件")
        sys.exit(1)

    print(f"找到 {len(shots)} 个镜头文件")
    output_dir = folder / "analysis_output"
    output_dir.mkdir(exist_ok=True)

    # 4. 组装 hint
    hint = {"product": args.product}
    if args.action:
        hint["action"] = args.action
    if args.audience:
        hint["audience"] = args.audience
    if args.hypotheses:
        hint["hypotheses"] = [h.strip() for h in args.hypotheses.split(",") if h.strip()]

    # 决定用哪个模型
    model_name = "google/gemini-2.5-pro" if args.pro else args.model

    print(f"人工提示: {hint}")
    print(f"模型: {model_name}")
    print(f"输出目录: {output_dir}\n")

    # 5. 逐个处理
    success = 0
    skip = 0
    fail = 0

    for i, shot_path in enumerate(shots, 1):
        shot_id = shot_path.stem
        out_file = output_dir / f"{shot_id}.json"

        prefix = f"[{i:03d}/{len(shots)}] {shot_id}"
        if out_file.exists() and not args.overwrite:
            print(f"{prefix} - 跳过（已存在，加 --overwrite 覆盖）")
            skip += 1
            continue

        try:
            file_mb = shot_path.stat().st_size / 1024 / 1024
            use_native = USE_NATIVE_VIDEO and (not args.frames) and (file_mb <= MAX_VIDEO_MB)

            if use_native:
                video_bytes = shot_path.read_bytes()
                duration = get_video_duration(shot_path)
                if duration <= 0:
                    print(f"{prefix} - 读取时长失败")
                    fail += 1
                    continue
                print(f"{prefix} - 原生视频 {file_mb:.1f}MB, {duration:.2f}s, 调用模型...")
                result = analyze_shot(
                    shot_id=shot_id,
                    duration=duration,
                    video_bytes=video_bytes,
                    human_hint=hint,
                    model=model_name,
                )
            else:
                keyframes, duration = extract_keyframes(shot_path, n_frames=args.n_frames)
                if not keyframes:
                    print(f"{prefix} - 抽帧失败")
                    fail += 1
                    continue
                reason = "强制关键帧" if args.frames else (f"视频 {file_mb:.1f}MB 超限，降级抽帧" if file_mb > MAX_VIDEO_MB else "关键帧模式")
                print(f"{prefix} - {reason}: 抽 {len(keyframes)} 帧, {duration:.2f}s, 调用模型...")
                result = analyze_shot(
                    shot_id=shot_id,
                    duration=duration,
                    keyframes=keyframes,
                    human_hint=hint,
                    model=model_name,
                )
            result = validate_evidence(result)

            out_file.write_text(
                result.model_dump_json(indent=2),
                encoding="utf-8",
            )

            n_sp = len(result.selling_points)
            n_co = len(result.coactive_events)
            print(f"{prefix} - ✓ 卖点 {n_sp} 条 / 协同 {n_co} 条 → {out_file.name}")
            success += 1

        except Exception as e:
            print(f"{prefix} - ✗ 失败: {e}")
            traceback.print_exc()
            fail += 1

    print(f"\n完成: 成功 {success} / 跳过 {skip} / 失败 {fail}")

    # 6. 整合级 JSON（商业化交付格式：短标签、去重、面向质检）
    if not args.no_aggregate:
        try:
            aggregate_path = aggregate_shots(output_dir, ad_id=folder.name, model=model_name)
            print(f"整合 JSON → {aggregate_path}")
        except Exception as e:
            print(f"[警告] 整合失败: {e}")
            traceback.print_exc()


# ============================================================
# 整合级 JSON：把每个 shot 的相对时间戳换算成整段广告的全局时间
# ============================================================

def _shift_ts(pair, offset):
    return [round(pair[0] + offset, 3), round(pair[1] + offset, 3)]


def aggregate_shots(output_dir: Path, ad_id: str, model: str = "google/gemini-2.5-flash") -> Path:
    """
    读取 output_dir/shot_*.json，先做时间戳全局化，然后送 Gemini 汇总成商业化交付 JSON。
    输出：ad_aggregate.json（面向人工质检的短标签格式）
    """
    files = sorted(output_dir.glob("shot_*.json"))
    if not files:
        raise RuntimeError(f"未找到分析结果文件: {output_dir}/shot_*.json")

    shots_global = []
    cursor = 0.0

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        local_dur = float(data.get("end_ts", 0.0)) - float(data.get("start_ts", 0.0))
        offset = cursor

        chains = data.get("action_chains", {}) or {}
        for a in chains.get("human", []):
            a["ts"] = _shift_ts(a.get("ts", [0, 0]), offset)
        for b in chains.get("product", []):
            b["ts"] = _shift_ts(b.get("ts", [0, 0]), offset)
        for c in chains.get("camera", []):
            c["ts"] = _shift_ts(c.get("ts", [0, 0]), offset)
        for o in chains.get("overlay", []):
            o["ts"] = _shift_ts(o.get("ts", [0, 0]), offset)

        data["start_ts"] = round(offset, 3)
        data["end_ts"] = round(offset + local_dur, 3)

        shots_global.append(data)
        cursor += local_dur

    # 调 Gemini 汇总成商业化格式
    result = summarize_to_business_json(
        shots=shots_global,
        ad_id=ad_id,
        total_duration=cursor,
        model=model,
    )

    out_path = output_dir / "ad_aggregate.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))
    return out_path


if __name__ == "__main__":
    main()
