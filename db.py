"""
任务记录（Postgres）
====================

记录每次 split / analyze 任务：类型、项目、状态、耗时、产物前缀、错误。
DATABASE_URL 未配置时全部降级为 no-op，不影响服务运行（本地/未接库时安全）。
Railway 加 Postgres 后会自动注入 DATABASE_URL。
"""

from __future__ import annotations
import os

_DSN = os.environ.get("DATABASE_URL", "")


def enabled() -> bool:
    return bool(_DSN)


def _conn():
    import psycopg
    return psycopg.connect(_DSN)


def init_db():
    """建表（幂等）。DATABASE_URL 没配就跳过。"""
    if not enabled():
        print("[db] DATABASE_URL 未配置，跳过（任务不落库）")
        return
    try:
        with _conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id           SERIAL PRIMARY KEY,
                    kind         TEXT,          -- split / analyze
                    project_id   TEXT,
                    ref          TEXT,          -- ad_id 或 product
                    status       TEXT,          -- done / error
                    ok           BOOLEAN,
                    r2_prefix    TEXT,
                    detail       TEXT,          -- 错误信息或备注
                    duration_sec REAL,
                    created_at   TIMESTAMPTZ DEFAULT now()
                )
            """)
        print("[db] jobs 表就绪")
        # 服务重启说明之前的 running 任务已随进程消失，标记为中断，避免永远挂着
        with _conn() as c:
            cur = c.execute(
                "UPDATE jobs SET status='error', ok=false,"
                " detail='服务重启导致任务中断' WHERE status='running'")
            n = cur.rowcount
        if n:
            print(f"[db] 已把 {n} 条孤儿 running 任务标记为中断")
    except Exception as e:
        print(f"[db] init 失败: {e}")


def start_job(kind: str, project_id: str, ref: str):
    """创建一条 running 任务，返回 id（供后台完成时更新）。"""
    if not enabled():
        return None
    try:
        with _conn() as c:
            cur = c.execute(
                "INSERT INTO jobs (kind, project_id, ref, status, ok) VALUES (%s,%s,%s,'running',NULL) RETURNING id",
                (kind, project_id, ref),
            )
            return cur.fetchone()[0]
    except Exception as e:
        print(f"[db] start_job 失败: {e}")
        return None


def finish_job(job_id, ok: bool, r2_prefix: str = "", detail: str = "", duration_sec: float = 0.0):
    """更新任务为完成/失败。"""
    if not enabled() or job_id is None:
        return
    try:
        with _conn() as c:
            c.execute(
                "UPDATE jobs SET status=%s, ok=%s, r2_prefix=%s, detail=%s, duration_sec=%s WHERE id=%s",
                ("done" if ok else "error", ok, r2_prefix, (detail or "")[:2000],
                 round(duration_sec, 2), job_id),
            )
    except Exception as e:
        print(f"[db] finish_job 失败: {e}")


def record(kind: str, project_id: str, ref: str, ok: bool,
           r2_prefix: str = "", detail: str = "", duration_sec: float = 0.0):
    """写一条任务记录。失败只打印、不抛出（不能因为记录失败影响主流程）。"""
    if not enabled():
        return None
    try:
        with _conn() as c:
            cur = c.execute(
                "INSERT INTO jobs (kind, project_id, ref, status, ok, r2_prefix, detail, duration_sec)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (kind, project_id, ref, "done" if ok else "error", ok,
                 r2_prefix, (detail or "")[:2000], round(duration_sec, 2)),
            )
            return cur.fetchone()[0]
    except Exception as e:
        print(f"[db] record 失败: {e}")
        return None


def recent(limit: int = 50):
    if not enabled():
        return []
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT id, kind, project_id, ref, status, ok, r2_prefix, duration_sec, created_at, detail"
                " FROM jobs ORDER BY id DESC LIMIT %s", (limit,)
            ).fetchall()
        cols = ["id", "kind", "project_id", "ref", "status", "ok",
                "r2_prefix", "duration_sec", "created_at", "detail"]
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            d["created_at"] = str(d["created_at"])
            out.append(d)
        return out
    except Exception as e:
        print(f"[db] recent 失败: {e}")
        return []
