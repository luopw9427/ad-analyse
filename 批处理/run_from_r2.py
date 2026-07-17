"""
从 R2 驱动的批量分析包装器
==========================

把"手动解压文件夹"换成"从 R2 拉取 → 调用现有 batch_analyze → 产物写回 R2"。
不改动已跑通的 batch_analyze.py，只在外层做 R2 同步。

流程：
  1. 从 R2 的 projects/{PROJECT_ID}/ 拉下所有 shot_*.mp4（每条广告一个子文件夹）
  2. 调 batch_analyze.py 在本地临时目录跑分析（合并/拆分流程）
  3. 把分析产物（analysis_output/**、共性镜头prompt.json、横向对比.md）写回 R2 同前缀

用法：
  python 批处理/run_from_r2.py --project test --product "章鱼"
  # 其余参数原样透传给 batch_analyze，如 --pro / --pipeline split / --overwrite

依赖：
  pip install boto3 python-dotenv
  凭证走项目根 .env（本地）或系统环境变量（Railway）
"""

from __future__ import annotations
import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

# 让本文件能 import 项目根的 r2_storage
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import r2_storage as r2  # noqa: E402

BATCH_ANALYZE = Path(__file__).resolve().parent / "batch_analyze.py"

# 这些后缀是"拆分阶段的输入/产物"，回传时跳过，避免把大 mp4 重复上传
SKIP_UPLOAD_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def sync_down(project_id: str, dest_root: Path) -> list[str]:
    """把 R2 projects/{project_id}/ 下所有对象拉到本地 dest_root，保持 ad_id/文件名 结构。"""
    prefix = f"projects/{project_id}/"
    keys = r2.list_keys(prefix)
    if not keys:
        raise RuntimeError(f"R2 上 {prefix} 没有任何对象，先在 Colab 跑拆分并上传")
    ads = set()
    for k in keys:
        rel = k[len(prefix):]            # 形如 章鱼/shot_000.mp4
        if not rel:
            continue
        local = dest_root / rel
        r2.download_file(k, local)
        ads.add(rel.split("/")[0])
    print(f"[R2↓] 从 {prefix} 拉下 {len(keys)} 个对象，{len(ads)} 条广告：{', '.join(sorted(ads))}")
    return sorted(ads)


def sync_up(project_id: str, src_root: Path) -> int:
    """把本地分析产物写回 R2（跳过 mp4 等拆分素材）。"""
    prefix = f"projects/{project_id}"
    n = 0
    for p in src_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in SKIP_UPLOAD_SUFFIXES:
            continue
        rel = p.relative_to(src_root).as_posix()
        r2.upload_file(f"{prefix}/{rel}", p)
        n += 1
    print(f"[R2↑] 回传 {n} 个分析产物到 {prefix}/")
    return n


def main():
    ap = argparse.ArgumentParser(description="从 R2 驱动的批量分析")
    ap.add_argument("--project", required=True, help="PROJECT_ID（与 Colab 上传时一致）")
    ap.add_argument("--product", required=True, help="产品名（透传给 batch_analyze）")
    ap.add_argument("--workdir", default=None, help="本地临时工作目录（默认系统临时目录）")
    ap.add_argument("--keep", action="store_true", help="跑完保留本地临时目录（默认删除）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只验证 R2 拉取是否通（列出并下载切片后停止，不跑分析、不花钱）")
    ap.add_argument("--resume-upload", action="store_true",
                    help="跳过拉取与分析，只把 --workdir 里已有的产物补传回 R2（回传中断后用）")
    # 其余未知参数原样透传给 batch_analyze
    args, passthrough = ap.parse_known_args()

    # 断点补传：只回传，不重新分析、不重新花钱
    if args.resume_upload:
        if not args.workdir:
            print("[错误] --resume-upload 需要配合 --workdir 指定之前保留的目录")
            sys.exit(1)
        root = Path(args.workdir) / args.project
        if not root.is_dir():
            print(f"[错误] 找不到 {root}")
            sys.exit(1)
        sync_up(args.project, root)
        print(f"\nOK 补传完成。R2 前缀：projects/{args.project}/")
        return

    # 连通性自检：只拉取、不分析、不回传
    if args.dry_run:
        check_root = Path(tempfile.mkdtemp(prefix=f"r2check_{args.project}_"))
        try:
            ads = sync_down(args.project, check_root / args.project)
            print(f"\nOK R2 拉取正常，共 {len(ads)} 条广告可分析。加 --dry-run 之外的参数即可正式跑。")
        finally:
            import shutil
            shutil.rmtree(check_root, ignore_errors=True)
        return

    # 工作目录
    if args.workdir:
        work = Path(args.workdir)
        work.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work = Path(tempfile.mkdtemp(prefix=f"r2_{args.project}_"))
        cleanup = not args.keep

    root = work / args.project
    root.mkdir(parents=True, exist_ok=True)

    ok = False
    try:
        # 1. 拉取
        sync_down(args.project, root)

        # 2. 调 batch_analyze
        cmd = [sys.executable, str(BATCH_ANALYZE), str(root),
               "--product", args.product] + passthrough
        print(f"[分析] {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if proc.returncode != 0:
            print(f"[警告] batch_analyze 返回码 {proc.returncode}，仍尝试回传已生成的产物")

        # 3. 回传
        sync_up(args.project, root)
        ok = True
        print(f"\nOK 完成。R2 前缀：projects/{args.project}/")
    finally:
        # 只有全部成功才清理；失败时保留本地产物，避免白跑一次分析
        if ok and cleanup:
            import shutil
            shutil.rmtree(work, ignore_errors=True)
            print(f"[清理] 已删除临时目录 {work}")
        else:
            print(f"[保留] 本地产物在：{root}")
            if not ok:
                print(f"      回传未完成。修好网络后可只补传：")
                print(f"      python 批处理/run_from_r2.py --project {args.project} "
                      f"--product \"{args.product}\" --resume-upload --workdir \"{work}\"")


if __name__ == "__main__":
    main()
