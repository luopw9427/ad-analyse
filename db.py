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
            # 已存在的表补列（幂等）
            c.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS engine TEXT")
            c.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS product TEXT")
            c.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS cost_usd REAL DEFAULT 0")
            c.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS tokens_in INTEGER DEFAULT 0")
            c.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS tokens_out INTEGER DEFAULT 0")
            c.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS cost_exact BOOLEAN DEFAULT true")
            c.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS batch TEXT")
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


def start_job(kind: str, project_id: str, ref: str, engine: str = "", product: str = "", batch: str = ""):
    """创建一条 running 任务，返回 id（供后台完成时更新）。"""
    if not enabled():
        return None
    try:
        with _conn() as c:
            cur = c.execute(
                "INSERT INTO jobs (kind, project_id, ref, status, ok, engine, product, batch)"
                " VALUES (%s,%s,%s,'running',NULL,%s,%s,%s) RETURNING id",
                (kind, project_id, ref, engine, product, batch),
            )
            return cur.fetchone()[0]
    except Exception as e:
        print(f"[db] start_job 失败: {e}")
        return None


def finish_job(job_id, ok: bool, r2_prefix: str = "", detail: str = "", duration_sec: float = 0.0,
               cost_usd: float = 0.0, tokens_in: int = 0, tokens_out: int = 0,
               cost_exact: bool = True):
    """更新任务为完成/失败。"""
    if not enabled() or job_id is None:
        return
    try:
        with _conn() as c:
            c.execute(
                "UPDATE jobs SET status=%s, ok=%s, r2_prefix=%s, detail=%s, duration_sec=%s,"
                " cost_usd=%s, tokens_in=%s, tokens_out=%s, cost_exact=%s WHERE id=%s",
                ("done" if ok else "error", ok, r2_prefix, (detail or "")[:2000],
                 round(duration_sec, 2), round(cost_usd, 6), tokens_in, tokens_out,
                 cost_exact, job_id),
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


def delete_job(job_id: int) -> int:
    """删除单条任务记录。"""
    if not enabled():
        return 0
    try:
        with _conn() as c:
            cur = c.execute("DELETE FROM jobs WHERE id=%s", (job_id,))
            return cur.rowcount
    except Exception as e:
        print(f"[db] delete_job 失败: {e}")
        return 0


def delete_jobs(project_id: str = "", batch: str = "") -> int:
    """删除任务记录：给了 batch 只删该批次；给了 project_id 删该项目；都没给清空。"""
    if not enabled():
        return 0
    try:
        with _conn() as c:
            if project_id and batch:
                cur = c.execute("DELETE FROM jobs WHERE project_id=%s AND batch=%s", (project_id, batch))
            elif project_id:
                cur = c.execute("DELETE FROM jobs WHERE project_id=%s", (project_id,))
            else:
                cur = c.execute("DELETE FROM jobs")
            n = cur.rowcount
            # 若整表已空，重置自增 id 从 1 开始
            left = c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            if left == 0:
                try:
                    c.execute("ALTER SEQUENCE jobs_id_seq RESTART WITH 1")
                except Exception:
                    pass
            return n
    except Exception as e:
        print(f"[db] delete_jobs 失败: {e}")
        return 0


def recent(limit: int = 50):
    if not enabled():
        return []
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT id, kind, project_id, ref, status, ok, r2_prefix, duration_sec, created_at, detail, engine, product, cost_usd, tokens_in, tokens_out, cost_exact, batch"
                " FROM jobs ORDER BY id DESC LIMIT %s", (limit,)
            ).fetchall()
        cols = ["id", "kind", "project_id", "ref", "status", "ok",
                "r2_prefix", "duration_sec", "created_at", "detail", "engine", "product",
                "cost_usd", "tokens_in", "tokens_out", "cost_exact", "batch"]
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            d["created_at"] = str(d["created_at"])
            out.append(d)
        return out
    except Exception as e:
        print(f"[db] recent 失败: {e}")
        return []
