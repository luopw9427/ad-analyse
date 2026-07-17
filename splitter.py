"""
视频镜头拆分（云端 CPU 版）
==========================

抽自 video_split notebook：TransNetV2 检测切点 + ffmpeg 切片。
在 Railway（无 GPU）上运行，import 前先 patch 掉 CUDA 调用，模型加载到 CPU。
模型按需懒加载（首次调用 /split 时才载入），不拖慢只用 /analyze 的场景。

产物：out_dir/shot_XXX.mp4 + out_dir/shots.json（不生成曲线图，服务端从简）。
"""

from __future__ import annotations
import os
import json
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np

_model = None
_DEVICE = None


def _ensure_model():
    """懒加载 TransNetV2 模型（CPU/GPU 自适应）。"""
    global _model, _DEVICE
    if _model is not None:
        return
    import torch

    if not torch.cuda.is_available():
        # 无 GPU：patch 掉 CUDA 调用，让 transnetv2pt import 时不崩
        torch.cuda.is_available = lambda: False
        torch.Tensor.cuda = lambda self, *a, **k: self
        torch.nn.Module.cuda = lambda self, *a, **k: self
        _orig_load = torch.load
        def _safe_load(*a, **k):
            k.setdefault("map_location", "cpu")
            return _orig_load(*a, **k)
        torch.load = _safe_load
        _DEVICE = "cpu"
    else:
        _DEVICE = "cuda"

    from transnetv2pt.inference import model
    _model = model.to(_DEVICE).eval()


def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """逐帧切点概率 → [(start_frame, end_frame), ...]。"""
    predictions = (predictions > threshold).astype(np.uint8)
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    if t == 0:
        scenes.append([start, i])
    return np.array(scenes, dtype=np.int32)


def _detect(video_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """用 ffmpeg 抽 48x27 帧 → predict_raw → scenes + 每帧概率。"""
    import ffmpeg
    from transnetv2pt.inference import predict_raw

    vs, _ = (
        ffmpeg.input(video_path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
        .run(capture_stdout=True, capture_stderr=True)
    )
    frames = np.frombuffer(vs, np.uint8).reshape([-1, 27, 48, 3])
    _, single_pred, _ = predict_raw(_model, frames)
    single_pred = np.asarray(single_pred).flatten()
    scenes = predictions_to_scenes(single_pred, threshold=0.5)
    return scenes, single_pred


def _fps(video_path: str) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", video_path,
    ]).decode().strip()
    n, d = out.split("/")
    return float(n) / float(d)


def split_video(video_path: str, out_dir: str, precise: bool = True,
                min_duration: float = 0.1) -> List[dict]:
    """
    拆分一个视频 → out_dir 下 shot_XXX.mp4 + shots.json。
    返回 shots 列表。
    """
    _ensure_model()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = _fps(video_path)
    scenes, _ = _detect(video_path)
    if scenes.ndim == 1:
        scenes = scenes.reshape(1, 2)

    shots = [{
        "idx": i,
        "start_frame": int(s), "end_frame": int(e),
        "start_time": round(s / fps, 3),
        "end_time": round(e / fps, 3),
        "duration": round((e - s) / fps, 3),
    } for i, (s, e) in enumerate(scenes)]

    # 过滤零长度/极短尾镜头 + 重新编号
    shots = [sh for sh in shots if sh["duration"] > min_duration]
    for new_idx, sh in enumerate(shots):
        sh["idx"] = new_idx

    with open(out_dir / "shots.json", "w", encoding="utf-8") as f:
        json.dump(shots, f, indent=2, ensure_ascii=False)

    for sh in shots:
        out = out_dir / f"shot_{sh['idx']:03d}.mp4"
        if precise:
            cmd = (f'ffmpeg -y -loglevel error -ss {sh["start_time"]} -to {sh["end_time"]} '
                   f'-i "{video_path}" -c:v libx264 -preset veryfast -c:a aac "{out}"')
        else:
            cmd = (f'ffmpeg -y -loglevel error -ss {sh["start_time"]} -to {sh["end_time"]} '
                   f'-i "{video_path}" -c copy "{out}"')
        os.system(cmd)

    return shots
