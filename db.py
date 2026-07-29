"""
任务记录（Postgres）
====================

记录每次 split / analyze 任务：类型、分组、状态、耗时、产物前缀、错误。
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
    """校验密码。兼容旧版单 sha256（迁移期用）。"""
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
            # 素材生成：任务谱系（改词重生成形成版本链）
            c.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS series_id TEXT")   # 同一谱系（一次创作的多版）
            c.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS parent_id INTEGER") # 上一版任务 id
            c.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS ver INTEGER")       # 谱系内第几版
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
            c.execute("ALTER TABLE fav_users ADD COLUMN IF NOT EXISTS credits INTEGER DEFAULT 0")  # 素材生成积分
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
            # 审计日志（登录/管理/删除等安全事件）
            c.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id       SERIAL PRIMARY KEY,
                    ts       TIMESTAMPTZ DEFAULT now(),
                    username TEXT,
                    ip       TEXT,
                    action   TEXT,
                    detail   TEXT
                )
            """)
        print("[db] jobs / favorites / audit_log 表就绪")
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


def delete_jobs(project_id: str = "", batch: str = "", owner: str = "") -> int:
    """删除任务记录：给了 owner 只删该用户；给了 batch 只删该批次；给了 project_id 删该分组；都没给清空。"""
    if not enabled():
        return 0
    try:
        with _conn() as c:
            conds, args = [], []
            if project_id:
                conds.append("project_id=%s"); args.append(project_id)
            if batch:
                conds.append("batch=%s"); args.append(batch)
            if owner:
                conds.append("owner=%s"); args.append(owner)
            if conds:
                cur = c.execute("DELETE FROM jobs WHERE " + " AND ".join(conds), tuple(args))
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


# ── 素材生成：积分 ────────────────────────────────────────────────
def user_credits(username: str) -> int:
    """查用户剩余积分。"""
    if not enabled():
        return 0
    try:
        with _conn() as c:
            r = c.execute("SELECT COALESCE(credits,0) FROM fav_users WHERE username=%s",
                          (username,)).fetchone()
        return int(r[0]) if r else 0
    except Exception as e:
        print(f"[db] user_credits 失败: {e}")
        return 0


def user_add_credits(username: str, delta: int) -> int:
    """充值/调整积分（delta 可负），返回调整后余额；用户不存在返回 -1。"""
    if not enabled():
        return -1
    try:
        with _conn() as c:
            r = c.execute(
                "UPDATE fav_users SET credits=GREATEST(0, COALESCE(credits,0)+%s)"
                " WHERE username=%s RETURNING credits", (int(delta), username)).fetchone()
        return int(r[0]) if r else -1
    except Exception as e:
        print(f"[db] user_add_credits 失败: {e}")
        return -1


def user_try_deduct(username: str, amount: int) -> bool:
    """原子扣减：余额足够才扣并返回 True，否则不动返回 False。"""
    if not enabled():
        return True   # 未接数据库时不拦截（本地/降级）
    if amount <= 0:
        return True
    try:
        with _conn() as c:
            r = c.execute(
                "UPDATE fav_users SET credits=COALESCE(credits,0)-%s"
                " WHERE username=%s AND COALESCE(credits,0)>=%s RETURNING credits",
                (int(amount), username, int(amount))).fetchone()
        return r is not None
    except Exception as e:
        print(f"[db] user_try_deduct 失败: {e}")
        return False


# ── 素材生成：任务查询 ────────────────────────────────────────────
_GEN_COLS = ("id, kind, project_id, ref, status, ok, r2_prefix, duration_sec,"
             " created_at, detail, cost_usd, batch, owner, series_id, parent_id, ver")
_GEN_KEYS = ["id", "kind", "project_id", "ref", "status", "ok", "r2_prefix",
             "duration_sec", "created_at", "detail", "cost_usd", "batch", "owner",
             "series_id", "parent_id", "ver"]


def _gen_row(r):
    d = dict(zip(_GEN_KEYS, r))
    d["created_at"] = str(d["created_at"])
    return d


def gen_list(owner: str = "", limit: int = 60, latest_only: bool = True):
    """列出生成任务（kind in gen_video/gen_image）。给了 owner 只看该用户，否则全体。
    latest_only=True 时每个谱系只返回最新一版（列表更清爽；详情里再展开历史版本）。"""
    if not enabled():
        return []
    try:
        with _conn() as c:
            sql = f"SELECT {_GEN_COLS} FROM jobs WHERE kind IN ('gen_video','gen_image')"
            args = []
            if owner:
                sql += " AND owner=%s"; args.append(owner)
            sql += " ORDER BY id DESC LIMIT %s"; args.append(max(limit * 4, 200))
            rows = c.execute(sql, tuple(args)).fetchall()
        out = [_gen_row(r) for r in rows]
        if latest_only:
            seen, filtered = set(), []
            for d in out:   # 已按 id DESC，先见到的即该谱系最新版
                sid = d.get("series_id") or f"job{d['id']}"
                if sid in seen:
                    continue
                seen.add(sid)
                filtered.append(d)
            out = filtered
        return out[:limit]
    except Exception as e:
        print(f"[db] gen_list 失败: {e}")
        return []


def gen_series(series_id: str):
    """列出同一谱系的所有版本（含改词重生成的历史），按版本升序。"""
    if not enabled() or not series_id:
        return []
    try:
        with _conn() as c:
            rows = c.execute(
                f"SELECT {_GEN_COLS} FROM jobs WHERE series_id=%s ORDER BY id ASC",
                (series_id,)).fetchall()
        return [_gen_row(r) for r in rows]
    except Exception as e:
        print(f"[db] gen_series 失败: {e}")
        return []


def gen_get(job_id: int):
    if not enabled():
        return None
    try:
        with _conn() as c:
            r = c.execute(
                f"SELECT {_GEN_COLS} FROM jobs WHERE id=%s AND kind IN ('gen_video','gen_image')",
                (job_id,)).fetchone()
        return _gen_row(r) if r else None
    except Exception as e:
        print(f"[db] gen_get 失败: {e}")
        return None


def gen_set_lineage(job_id, series_id: str, parent_id, ver: int):
    """给刚创建的生成任务写入谱系信息。series_id 为空时用 job 自身 id 兜底。"""
    if not enabled() or job_id is None:
        return
    try:
        sid = series_id or f"s{job_id}"
        with _conn() as c:
            c.execute("UPDATE jobs SET series_id=%s, parent_id=%s, ver=%s WHERE id=%s",
                      (sid, parent_id, int(ver or 1), job_id))
    except Exception as e:
        print(f"[db] gen_set_lineage 失败: {e}")


def gen_series_count(series_id: str) -> int:
    if not enabled() or not series_id:
        return 0
    try:
        with _conn() as c:
            r = c.execute("SELECT COUNT(*) FROM jobs WHERE series_id=%s", (series_id,)).fetchone()
        return int(r[0]) if r else 0
    except Exception:
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
    """预置管理员：不存在则以 admin 建；已存在则确保其 role=admin（不改密码）。"""
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
    """校验用户名+密码。成功返回用户 dict（不含 pin_hash），并把旧格式密码透明升级为 PBKDF2；失败返回 None。"""
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
                "SELECT username, role, disabled, created_at, COALESCE(credits,0)"
                " FROM fav_users ORDER BY username"
            ).fetchall()
        return [{"username": r[0], "role": r[1] or "user", "disabled": bool(r[2]),
                 "created_at": str(r[3]), "credits": int(r[4])} for r in rows]
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
    """管理员重置密码：改哈希并 pin_ver+1（旧 token 失效）。"""
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
    """用户自助改密码：校验旧密码，改哈希并 pin_ver+1。"""
    u = user_get(username)
    if not u:
        return False, "用户不存在"
    if not _pin_verify(old_pin, u["pin_hash"]):
        return False, "原密码不正确"
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


# ── 审计日志 + 数据库查看（管理员后台）─────────────────────────
def audit(username: str, ip: str, action: str, detail: str = ""):
    """记录一条安全审计。失败只打印，不影响主流程。"""
    if not enabled():
        return
    try:
        with _conn() as c:
            c.execute("INSERT INTO audit_log (username, ip, action, detail) VALUES (%s,%s,%s,%s)",
                      (username or "", ip or "", action or "", (detail or "")[:1000]))
    except Exception as e:
        print(f"[db] audit 失败: {e}")


def audit_has_login(username: str) -> bool:
    """该用户是否已有过成功登录记录（用于区分"首次登录"）。"""
    if not enabled() or not username:
        return False
    try:
        with _conn() as c:
            r = c.execute("SELECT 1 FROM audit_log WHERE username=%s AND action IN ('login','login_first') LIMIT 1",
                          (username,)).fetchone()
        return bool(r)
    except Exception:
        return False


def audit_list(limit: int = 200, username: str = "", action: str = ""):
    if not enabled():
        return []
    try:
        where, args = [], []
        if username:
            where.append("username=%s"); args.append(username)
        if action:
            where.append("action=%s"); args.append(action)
        sql = "SELECT id, ts, username, ip, action, detail FROM audit_log"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT %s"; args.append(limit)
        with _conn() as c:
            rows = c.execute(sql, tuple(args)).fetchall()
        return [{"id": r[0], "ts": str(r[1]), "username": r[2], "ip": r[3],
                 "action": r[4], "detail": r[5]} for r in rows]
    except Exception as e:
        print(f"[db] audit_list 失败: {e}")
        return []


# 只读表查看白名单：逻辑名 -> (真实表, 列, 排序)
_TABLES = {
    "users":     ("fav_users", ["username", "role", "disabled", "created_at"], "username ASC"),
    "jobs":      ("jobs", ["id", "kind", "project_id", "batch", "product", "owner", "status", "duration_sec", "cost_usd", "created_at"], "id DESC"),
    "favorites": ("favorites", ["id", "owner", "kind", "ref", "label", "created_at"], "id DESC"),
    "audit":     ("audit_log", ["id", "ts", "username", "ip", "action", "detail"], "id DESC"),
}


# 各表可按"成员/分组"过滤的列
_FILTER_OWNER = {"jobs": "owner", "favorites": "owner", "audit": "username", "users": "username"}
_FILTER_PROJ = {"jobs": "project_id"}


def table_rows(name: str, limit: int = 200, owner: str = "", project: str = ""):
    """白名单表查看（含 id 便于删行）。可按成员/分组过滤。返回 {cols, rows}。"""
    if not enabled() or name not in _TABLES:
        return {"cols": [], "rows": []}
    tbl, cols, order = _TABLES[name]
    sel = cols if "id" in cols else (["id"] + cols) if tbl != "fav_users" else cols
    where, args = [], []
    if owner and name in _FILTER_OWNER:
        where.append(f"{_FILTER_OWNER[name]}=%s"); args.append(owner)
    if project and name in _FILTER_PROJ:
        where.append(f"{_FILTER_PROJ[name]}=%s"); args.append(project)
    try:
        sql = f"SELECT {', '.join(sel)} FROM {tbl}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {order} LIMIT %s"; args.append(limit)
        with _conn() as c:
            rows = c.execute(sql, tuple(args)).fetchall()
        out = [[("" if v is None else str(v)) for v in r] for r in rows]
        return {"cols": sel, "rows": out}
    except Exception as e:
        print(f"[db] table_rows 失败: {e}")
        return {"cols": [], "rows": []}


def table_delete_row(name: str, row_id) -> int:
    """删除白名单表的一行（按 id）。users 表请走用户管理接口。"""
    real = {"jobs": "jobs", "favorites": "favorites", "audit": "audit_log"}
    if not enabled() or name not in real:
        return 0
    try:
        with _conn() as c:
            cur = c.execute(f"DELETE FROM {real[name]} WHERE id=%s", (row_id,))
            return cur.rowcount
    except Exception as e:
        print(f"[db] table_delete_row 失败: {e}")
        return 0


def db_stats():
    """各表行数统计。"""
    if not enabled():
        return {}
    out = {}
    try:
        with _conn() as c:
            for name, (tbl, _, _) in _TABLES.items():
                try:
                    out[name] = c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                except Exception:
                    out[name] = 0
    except Exception as e:
        print(f"[db] db_stats 失败: {e}")
    return out


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
