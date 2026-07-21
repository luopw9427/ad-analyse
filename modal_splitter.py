"""
视频镜头拆分 · Modal GPU 版
===========================

在 Modal 的 T4 GPU 上跑 TransNetV2 检测切点 + ffmpeg 切片，直接读写 Cloudflare R2。
由 GitHub Actions 执行 `modal deploy` 部署（本地无需连 Modal）。

调用方式（Railway 侧）：
    POST <web_url>
    body: {"token":"...", "project_id":"test", "ad_id":"海马",
           "raw_key":"raw/test/海马.mp4", "precise":false}
    resp: {"ok":true, "n_shots":9, "shots":[...], "r2_prefix":"projects/test/海马/"}

依赖的 Modal Secret（在 Modal 网页后台创建）：
    r2-credentials : R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY /
                     R2_BUCKET_NAME / R2_ENDPOINT / R2_PUBLIC_URL
    split-auth     : SPLIT_TOKEN
"""

import modal

app = modal.App("video-split-gpu")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "numpy",
        "boto3",
        "ffmpeg-python",
        "fastapi[standard]",
        "torch",
        "git+https://github.com/kunato/transnetv2pt.git",
    )
)

r2_secret = modal.Secret.from_name("r2-credentials")
auth_secret = modal.Secret.from_name("split-auth")


@app.cls(
    image=image,
    gpu="T4",
    secrets=[r2_secret, auth_secret],
    scaledown_window=300,   # 空闲 5 分钟才缩容：批量拆分时只有首条有冷启动
    timeout=1800,
)
class Splitter:

    @modal.enter()
    def load_model(self):
        """容器启动时把 TransNetV2 加载到 GPU（此处 CUDA 可用，无需任何 patch）。"""
        import torch
        from transnetv2pt.inference import model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        print(f"[Modal] TransNetV2 已加载，设备={self.device}")

    # ---------- R2 ----------
    def _r2(self):
        import os
        import boto3
        from botocore.client import Config
        return boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            config=Config(signature_version="s3v4",
                          retries={"max_attempts": 5, "mode": "standard"}),
            region_name="auto",
        )

    # ---------- 切点检测 ----------
    @staticmethod
    def _predictions_to_scenes(predictions, threshold=0.5):
        import numpy as np
        predictions = (predictions > threshold).astype(np.uint8)
        scenes, t, t_prev, start = [], -1, 0, 0
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

    def _detect(self, video_path):
        import numpy as np
        import ffmpeg
        from transnetv2pt.inference import predict_raw
        vs, _ = (
            ffmpeg.input(video_path)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
            .run(capture_stdout=True, capture_stderr=True)
        )
        frames = np.frombuffer(vs, np.uint8).reshape([-1, 27, 48, 3])
        _, single_pred, _ = predict_raw(self.model, frames)
        single_pred = np.asarray(single_pred).flatten()
        return self._predictions_to_scenes(single_pred, 0.5)

    # ---------- HTTP 接口 ----------
    @modal.fastapi_endpoint(method="POST")
    def split(self, item: dict):
        import os
        import json
        import time
        import shutil
        import subprocess
        import tempfile
        from pathlib import Path
        from fastapi import HTTPException

        if item.get("token") != os.environ.get("SPLIT_TOKEN"):
            raise HTTPException(status_code=401, detail="invalid token")

        project_id = item["project_id"]
        ad_id = item["ad_id"]
        raw_key = item["raw_key"]
        precise = bool(item.get("precise", False))
        bucket = os.environ["R2_BUCKET_NAME"]

        t0 = time.time()
        s3 = self._r2()
        work = Path(tempfile.mkdtemp(prefix="split_"))
        try:
            # 1. 从 R2 取原视频
            video_path = work / "input.mp4"
            s3.download_file(bucket, raw_key, str(video_path))

            # 2. 帧率 + GPU 检测切点
            fps_out = subprocess.check_output([
                "ffprobe", "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate", str(video_path),
            ]).decode().strip()
            n, d = fps_out.split("/")
            fps = float(n) / float(d)

            scenes = self._detect(str(video_path))
            if scenes.ndim == 1:
                scenes = scenes.reshape(1, 2)

            shots = [{
                "idx": i,
                "start_frame": int(s), "end_frame": int(e),
                "start_time": round(s / fps, 3),
                "end_time": round(e / fps, 3),
                "duration": round((e - s) / fps, 3),
            } for i, (s, e) in enumerate(scenes)]
            shots = [sh for sh in shots if sh["duration"] > 0.1]
            for i, sh in enumerate(shots):
                sh["idx"] = i

            # 3. 切片
            out_dir = work / "out"
            out_dir.mkdir(exist_ok=True)
            for sh in shots:
                out = out_dir / f"shot_{sh['idx']:03d}.mp4"
                if precise:
                    cmd = (f'ffmpeg -y -loglevel error -ss {sh["start_time"]} -to {sh["end_time"]} '
                           f'-i "{video_path}" -c:v libx264 -preset veryfast -c:a aac "{out}"')
                else:
                    cmd = (f'ffmpeg -y -loglevel error -ss {sh["start_time"]} -to {sh["end_time"]} '
                           f'-i "{video_path}" -c copy "{out}"')
                os.system(cmd)

            (out_dir / "shots.json").write_text(
                json.dumps(shots, indent=2, ensure_ascii=False), encoding="utf-8")

            # 4. 回传 R2
            prefix = f"projects/{project_id}/{ad_id}/"
            uploaded = 0
            for fp in sorted(out_dir.iterdir()):
                ctype = "video/mp4" if fp.suffix == ".mp4" else "application/json"
                s3.upload_file(str(fp), bucket, prefix + fp.name,
                               ExtraArgs={"ContentType": ctype})
                uploaded += 1

            return {
                "ok": True, "n_shots": len(shots), "uploaded": uploaded,
                "r2_prefix": prefix, "shots": shots,
                "device": self.device, "elapsed": round(time.time() - t0, 2),
            }
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "elapsed": round(time.time() - t0, 2)}
        finally:
            shutil.rmtree(work, ignore_errors=True)
