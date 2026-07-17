"""
R2 对象存储工具（S3 兼容）
==========================

把视频 / 切片 / JSON 产物存到 Cloudflare R2，取代"本地文件夹 + zip 下载"。

依赖：
    pip install boto3 python-dotenv

配置（从环境变量读，切勿把密钥写死在代码里）：
    R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY
    R2_BUCKET_NAME / R2_ENDPOINT / R2_PUBLIC_URL
本地开发时可用 .env（见 .env.example），Railway 上在项目 Variables 里配。
"""

from __future__ import annotations
import os
import io
import json
import time
import mimetypes
from pathlib import Path
from typing import Optional, Union

import boto3
from botocore.client import Config

# 自动加载同目录/上层的 .env（本地开发用；Railway 上用系统环境变量，找不到也无妨）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        raise RuntimeError(f"缺少环境变量 {key}，请在 .env 或 Railway Variables 里配置")
    return v


def get_client():
    """创建 R2（S3 兼容）客户端。region 固定 'auto'。"""
    return boto3.client(
        "s3",
        endpoint_url=_env("R2_ENDPOINT"),
        aws_access_key_id=_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_env("R2_SECRET_ACCESS_KEY"),
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            connect_timeout=30,
            read_timeout=120,
        ),
        region_name="auto",
    )


def _retry(fn, *args, _tries: int = 6, _what: str = "R2 操作", **kwargs):
    """对付代理/VPN 偶发的 SSL EOF、连接重置、连接关闭：应用层再包一层重试。
    按异常类型名匹配，覆盖 botocore/urllib3/socket 各种瞬时网络错误。"""
    _transient_names = (
        "SSL", "Connection", "Timeout", "EndpointConnection",
        "ConnectionClosed", "ProtocolError", "IncompleteRead", "Reset",
    )
    last = None
    for i in range(1, _tries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            name = type(e).__name__
            is_transient = any(t in name for t in _transient_names)
            if not is_transient:
                raise
            last = e
            if i < _tries:
                wait = min(2 * i, 15)
                print(f"    [{_what}] 网络中断({name})第 {i}/{_tries} 次，{wait}s 后重试…")
                time.sleep(wait)
            continue
    raise RuntimeError(f"{_what} 重试 {_tries} 次仍失败（多为代理/VPN 不稳定，换节点或关代理再试）：{last}")


BUCKET = os.environ.get("R2_BUCKET_NAME", "")
PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")


# ============================================================
# 上传
# ============================================================

def upload_bytes(key: str, data: bytes, content_type: Optional[str] = None) -> str:
    """上传字节流，返回可访问的公开 URL。"""
    client = get_client()
    if content_type is None:
        content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    _retry(client.put_object, Bucket=BUCKET, Key=key, Body=data,
           ContentType=content_type, _what=f"上传 {key}")
    return public_url(key)


def upload_file(key: str, local_path: Union[str, Path]) -> str:
    """上传本地文件（如 shot_000.mp4）。"""
    client = get_client()
    ctype = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
    _retry(client.upload_file, str(local_path), BUCKET, key,
           ExtraArgs={"ContentType": ctype}, _what=f"上传 {key}")
    return public_url(key)


def upload_json(key: str, obj) -> str:
    """把 dict/list 序列化成 JSON 上传。"""
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    return upload_bytes(key, data, content_type="application/json; charset=utf-8")


def upload_dir(local_dir: Union[str, Path], key_prefix: str) -> list:
    """把整个目录（含子目录）上传到 key_prefix/ 下，返回所有 URL。"""
    local_dir = Path(local_dir)
    urls = []
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            urls.append(upload_file(f"{key_prefix}/{rel}", p))
    return urls


# ============================================================
# 下载 / 读取
# ============================================================

def download_bytes(key: str) -> bytes:
    client = get_client()
    buf = io.BytesIO()
    _retry(client.download_fileobj, BUCKET, key, buf, _what=f"下载 {key}")
    return buf.getvalue()


def download_file(key: str, local_path: Union[str, Path]) -> Path:
    client = get_client()
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _retry(client.download_file, BUCKET, key, str(local_path), _what=f"下载 {key}")
    return local_path


def read_json(key: str):
    return json.loads(download_bytes(key).decode("utf-8"))


def list_keys(prefix: str) -> list:
    """列出某前缀下所有对象 key（如某广告下的所有 shot）。"""
    client = get_client()
    keys, token = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = _retry(client.list_objects_v2, _what=f"列举 {prefix}", **kw)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]
    return keys


def public_url(key: str) -> str:
    """拼公开访问 URL（bucket 已开 r2.dev 公开访问时可用）。"""
    return f"{PUBLIC_URL}/{key}" if PUBLIC_URL else key


# ============================================================
# 自测：python r2_storage.py
# ============================================================

if __name__ == "__main__":
    test_key = "healthcheck/hello.txt"
    url = upload_bytes(test_key, b"hello r2", content_type="text/plain")
    print("上传成功：", url)
    print("读回内容：", download_bytes(test_key).decode())
    print("列目录：", list_keys("healthcheck/"))
