"""
任务记录（Postgres）
====================

记录每次 split / analyze 任务：类型、项目、状态、耗时、产物前缀、错误。
DATABASE_URL 未配置时全部降级为 no-op，不影响服务运行（本地/未接库时安全）。
Railway 加 Postgres 后会自动注入 DATABASE_URL。
"""

from __future__ import annotations
import os
import json as _json
import hashlib
import base64
import hmac

_DSN = os.environ.get("DATABASE_URL", "")
_PBKDF2_ITER = 200_000


def _pin_hash(pin: str) -> str:
    """PBKDF2-HMAC-SHA256 + 每人随机盐。格式 pbkdf2_sha256$迭代$盐$哈希。"""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", (pin or "").encode("utf-8"), salt, _PBKDF2_ITER)
    return "pbkdf2_sha256${}${}${}".format(
        _PBKDF2_ITER, base64.b64encode(salt).decode(), base64.b64encode(dk).decode())


def _pin_verify(pin: str, stored: str) -> bool:
    """校验口令。兼容旧版单 sha256（迁移期用）。"""
    if not stored:
        return False
    try:
        if stored.startswith("pbkdf2_sha256$"):
            _, it, salt_b64, hash_b64 = stored.split("$")
            dk = hashlib.pbkdf2_hmac("sha256", (pin or "").encode("utf-8"),
                                     base64.b64decode(salt_b64), int(it))
            return hmac.compare_digest(base64.b64encode(dk).decode(), hash_b64)
        # 旧格式：sha256("fav-salt:"+pin)
        legacy = hashlib.sha256(("fav-salt:" + (pin or "")).encode("utf-8")).hexdigest()
        return hmac.compare_digest(stored, legacy)
    except Exception:
        return False


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
            c.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS owner TEXT")   # 任务归属登录用户
            # 收藏 + 收藏用户（为将来登录预留）
            c.execute("""
                CREATE TABLE IF NOT EXISTS fav_users (
                    username   TEXT PRIMARY KEY,
                    pin_hash   TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            # 登录/权限扩展列
            c.execute("ALTER TABLE fav_users ADD COLUMN IF NOT EXISTS role TEXT DEFAULT 'user'")
            c.execute("ALTER TABLE fav_users ADD COLUMN IF NOT EXISTS pin_ver INTEGER DEFAULT 1")
            c.execute("ALTER TABLE fav_users ADD COLUMN IF NOT EXISTS disabled BOOLEAN DEFAULT false")
            c.execute("""
                CREATE TABLE IF NOT EXISTS favorites (
                    id         SERIAL PRIMARY KEY,
                    owner      TEXT NOT NULL,        -- 收藏归属用户名
                    kind       TEXT,                 -- task / clip / prompt
                    ref        TEXT,                 -- 去重键
                    label      TEXT,                 -- 展示名
                    meta       JSONB,                -- 渲染所需数据快照
                    created_at TIMESTAMPTZ DEFAULT now(),
                    UNIQUE (owner, ref)
                )
            """)
        print("[db] jobs / favorites 表就绪")
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


def start_job(kind: str, project_id: str, ref: str, engine: str = "", product: str = "", batch: str = "", owner: str = ""):
    """创建一条 running 任务，返回 id（供后台完成时更新）。"""
    if not enabled():
        return None
    try:
        with _conn() as c:
            cur = c.execute(
                "INSERT INTO jobs (kind, project_id, ref, status, ok, engine, product, batch, owner)"
                " VALUES (%s,%s,%s,'running',NULL,%s,%s,%s,%s) RETURNING id",
                (kind, project_id, ref, engine, product, batch, owner),
            )
            return cur.fetchone()[0]
    except Exception as e:
        print(f"[db] start_job 失败: {e}")
        return None


def job_owner(job_id: int):
    if not enabled():
        return None
    try:
        with _conn() as c:
            r = c.execute("SELECT owner FROM jobs WHERE id=%s", (job_id,)).fetchone()
        return (r[0] if r else None)
    except Exception:
        return None


def batch_owner(project_id: str, batch: str):
    """该批次的创建者（取拆分/分析任务的 owner）。找不到返回 None。"""
    if not enabled() or not batch:
        return None
    try:
        with _conn() as c:
            r = c.execute(
                "SELECT owner FROM jobs WHERE project_id=%s AND batch=%s AND owner IS NOT NULL"
                " ORDER BY id ASC LIMIT 1", (project_id, batch)).fetchone()
        return (r[0] if r else None)
    except Exception:
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


def delete_job(job_id: int, requester: str = "", is_admin: bool = False) -> int:
    """删除单条任务记录。非管理员只能删自己的（owner 为空的历史任务也允许删，兼容）。"""
    if not enabled():
        return 0
    try:
        with _conn() as c:
            if is_admin:
                cur = c.execute("DELETE FROM jobs WHERE id=%s", (job_id,))
            else:
                cur = c.execute(
                    "DELETE FROM jobs WHERE id=%s AND (owner=%s OR owner IS NULL OR owner='')",
                    (job_id, requester))
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


# ── 用户与登录 ───────────────────────────────────────────────────
def fav_user_ensure(username: str, pin: str = "", role: str = "user") -> bool:
    """确保用户存在（不存在则创建）。"""
    if not enabled() or not username:
        return False
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO fav_users (username, pin_hash, role) VALUES (%s,%s,%s)"
                " ON CONFLICT (username) DO NOTHING",
                (username, _pin_hash(pin), role),
            )
        return True
    except Exception as e:
        print(f"[db] fav_user_ensure 失败: {e}")
        return False


def user_ensure_admin(username: str, pin: str) -> bool:
    """预置管理员：不存在则以 admin 建；已存在则确保其 role=admin（不改口令）。"""
    if not enabled() or not username:
        return False
    try:
        with _conn() as c:
            row = c.execute("SELECT 1 FROM fav_users WHERE username=%s", (username,)).fetchone()
            if row:
                c.execute("UPDATE fav_users SET role='admin', disabled=false WHERE username=%s", (username,))
            else:
                c.execute("INSERT INTO fav_users (username, pin_hash, role) VALUES (%s,%s,'admin')",
                          (username, _pin_hash(pin)))
        print(f"[db] 预置管理员就绪: {username}")
        return True
    except Exception as e:
        print(f"[db] user_ensure_admin 失败: {e}")
        return False


def user_get(username: str):
    if not enabled() or not username:
        return None
    try:
        with _conn() as c:
            r = c.execute(
                "SELECT username, pin_hash, role, pin_ver, disabled FROM fav_users WHERE username=%s",
                (username,)).fetchone()
        if not r:
            return None
        return {"username": r[0], "pin_hash": r[1], "role": r[2] or "user",
                "pin_ver": int(r[3] or 1), "disabled": bool(r[4])}
    except Exception as e:
        print(f"[db] user_get 失败: {e}")
        return None


def user_verify(username: str, pin: str):
    """校验用户名+口令。成功返回用户 dict（不含 pin_hash），并把旧格式口令透明升级为 PBKDF2；失败返回 None。"""
    u = user_get(username)
    if not u or u["disabled"]:
        return None
    if not _pin_verify(pin, u["pin_hash"]):
        return None
    if not (u["pin_hash"] or "").startswith("pbkdf2_sha256$"):  # 透明升级旧哈希
        try:
            with _conn() as c:
                c.execute("UPDATE fav_users SET pin_hash=%s WHERE username=%s", (_pin_hash(pin), username))
        except Exception:
            pass
    return {"username": u["username"], "role": u["role"], "pin_ver": u["pin_ver"]}


def user_list():
    if not enabled():
        return []
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT username, role, disabled, created_at FROM fav_users ORDER BY username"
            ).fetchall()
        return [{"username": r[0], "role": r[1] or "user", "disabled": bool(r[2]),
                 "created_at": str(r[3])} for r in rows]
    except Exception as e:
        print(f"[db] user_list 失败: {e}")
        return []


def user_create(username: str, pin: str, role: str = "user") -> tuple:
    if not enabled():
        return False, "未接数据库"
    username = (username or "").strip()
    if not username:
        return False, "用户名不能为空"
    if role not in ("user", "admin"):
        role = "user"
    try:
        with _conn() as c:
            if c.execute("SELECT 1 FROM fav_users WHERE username=%s", (username,)).fetchone():
                return False, "用户名已存在"
            c.execute("INSERT INTO fav_users (username, pin_hash, role) VALUES (%s,%s,%s)",
                      (username, _pin_hash(pin), role))
        return True, "ok"
    except Exception as e:
        return False, str(e)


def user_set_role(username: str, role: str) -> tuple:
    if not enabled():
        return False, "未接数据库"
    if role not in ("user", "admin"):
        return False, "角色非法"
    try:
        with _conn() as c:
            cur = c.execute("UPDATE fav_users SET role=%s WHERE username=%s", (role, username))
            return (cur.rowcount > 0), ("ok" if cur.rowcount else "用户不存在")
    except Exception as e:
        return False, str(e)


def user_reset_pin(username: str, pin: str) -> tuple:
    """管理员重置口令：改哈希并 pin_ver+1（旧 token 失效）。"""
    if not enabled():
        return False, "未接数据库"
    try:
        with _conn() as c:
            cur = c.execute(
                "UPDATE fav_users SET pin_hash=%s, pin_ver=COALESCE(pin_ver,1)+1 WHERE username=%s",
                (_pin_hash(pin), username))
            return (cur.rowcount > 0), ("ok" if cur.rowcount else "用户不存在")
    except Exception as e:
        return False, str(e)


def user_change_pin(username: str, old_pin: str, new_pin: str) -> tuple:
    """用户自助改口令：校验旧口令，改哈希并 pin_ver+1。"""
    u = user_get(username)
    if not u:
        return False, "用户不存在"
    if not _pin_verify(old_pin, u["pin_hash"]):
        return False, "原口令不正确"
    try:
        with _conn() as c:
            c.execute("UPDATE fav_users SET pin_hash=%s, pin_ver=COALESCE(pin_ver,1)+1 WHERE username=%s",
                      (_pin_hash(new_pin), username))
        return True, "ok"
    except Exception as e:
        return False, str(e)


def user_set_disabled(username: str, disabled: bool) -> tuple:
    if not enabled():
        return False, "未接数据库"
    try:
        with _conn() as c:
            cur = c.execute("UPDATE fav_users SET disabled=%s WHERE username=%s", (disabled, username))
            return (cur.rowcount > 0), ("ok" if cur.rowcount else "用户不存在")
    except Exception as e:
        return False, str(e)


def user_delete(username: str) -> tuple:
    if not enabled():
        return False, "未接数据库"
    try:
        with _conn() as c:
            cur = c.execute("DELETE FROM fav_users WHERE username=%s", (username,))
            return (cur.rowcount > 0), ("ok" if cur.rowcount else "用户不存在")
    except Exception as e:
        return False, str(e)


def admin_count() -> int:
    if not enabled():
        return 0
    try:
        with _conn() as c:
            return c.execute("SELECT COUNT(*) FROM fav_users WHERE role='admin' AND disabled=false").fetchone()[0]
    except Exception:
        return 0


# ── 收藏（团队共享可见；增删见 app 层权限）─────────────────────────
def fav_add(owner: str, kind: str, ref: str, label: str = "", meta=None):
    """新增/更新收藏，返回该行 id。"""
    if not enabled() or not owner or not ref:
        return None
    try:
        with _conn() as c:
            cur = c.execute(
                "INSERT INTO favorites (owner, kind, ref, label, meta)"
                " VALUES (%s,%s,%s,%s,%s::jsonb)"
                " ON CONFLICT (owner, ref) DO UPDATE SET label=EXCLUDED.label, meta=EXCLUDED.meta"
                " RETURNING id",
                (owner, kind, ref, (label or "")[:500], _json.dumps(meta or {}, ensure_ascii=False)),
            )
            return cur.fetchone()[0]
    except Exception as e:
        print(f"[db] fav_add 失败: {e}")
        return None


def fav_remove_by_id(fav_id: int, requester: str = "", is_admin: bool = False) -> int:
    """按 id 删收藏：管理员可删任何人的，普通用户只能删自己的。"""
    if not enabled():
        return 0
    try:
        with _conn() as c:
            if is_admin:
                cur = c.execute("DELETE FROM favorites WHERE id=%s", (fav_id,))
            else:
                cur = c.execute("DELETE FROM favorites WHERE id=%s AND owner=%s", (fav_id, requester))
            return cur.rowcount
    except Exception as e:
        print(f"[db] fav_remove_by_id 失败: {e}")
        return 0


def fav_list_all():
    """所有人的收藏（团队共享可见），带 owner。"""
    if not enabled():
        return []
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT id, owner, kind, ref, label, meta, created_at FROM favorites ORDER BY id DESC"
            ).fetchall()
        return [{"id": r[0], "owner": r[1], "kind": r[2], "ref": r[3], "label": r[4],
                 "meta": r[5] or {}, "created_at": str(r[6])} for r in rows]
    except Exception as e:
        print(f"[db] fav_list_all 失败: {e}")
        return []


def recent(limit: int = 50):
    if not enabled():
        return []
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT id, kind, project_id, ref, status, ok, r2_prefix, duration_sec, created_at, detail, engine, product, cost_usd, tokens_in, tokens_out, cost_exact, batch, owner"
                " FROM jobs ORDER BY id DESC LIMIT %s", (limit,)
            ).fetchall()
        cols = ["id", "kind", "project_id", "ref", "status", "ok",
                "r2_prefix", "duration_sec", "created_at", "detail", "engine", "product",
                "cost_usd", "tokens_in", "tokens_out", "cost_exact", "batch", "owner"]
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            d["created_at"] = str(d["created_at"])
            out.append(d)
        return out
    except Exception as e:
        print(f"[db] recent 失败: {e}")
        return []
