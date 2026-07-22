"""
PostgresBackend
---------------
Redis-compatible storage backend using Postgres — the hosted-mode twin of
local_store.SQLiteBackend. Implements the exact same duck-typed method
surface (get/set/setex/delete/exists/expire/incr/rpush/lpush/llen/lrange/
ltrim/eval/pipeline/zadd/zrevrange/zrange/zremrangebyrank/zcard/hset/
hgetall/hincrby/scan_iter) so ProfileStore, MemoryStore, MagnetTeamStore,
SignalBuffer, EpisodicStore (redis-fallback path), and UsageCounter all work
completely unchanged on top of it.

Activated when MAGNET_DATABASE_URL is set and HTTP (hosted) mode is running
(see mcp_server._get_backend). Never used by default stdio mode.

Sync driver (psycopg v3 + psycopg_pool) is used deliberately: every existing
store method is a plain synchronous call, and mcp_server.py already wraps
each one in asyncio.to_thread(...). A sync driver keeps PostgresBackend a
literal drop-in for SQLiteBackend with zero changes to any call site.

This file also owns the Postgres connection pool and the (non-Redis-shaped)
relational tables used only by hosted-mode-specific code: api_keys (read by
magnet.auth) and usage_events (written by magnet.usage_counter). Those are
NOT part of the duck-typed surface below — callers that need them use
get_pool()/get_pool_if_configured() directly.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_pool: Any = None
_pool_lock = threading.Lock()


def get_pool(database_url: str | None = None) -> Any:
    """Shared psycopg_pool.ConnectionPool — created once per process."""
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        from psycopg_pool import ConnectionPool

        url = database_url or os.environ.get("MAGNET_DATABASE_URL")
        if not url:
            raise RuntimeError("MAGNET_DATABASE_URL is not set")
        _pool = ConnectionPool(url, min_size=1, max_size=10, open=True)
        logger.info("[postgres] connection pool created")
        return _pool


def get_pool_if_configured() -> Any | None:
    """Like get_pool(), but returns None instead of raising when Postgres
    isn't configured or unreachable — the safe entry point for code that
    must no-op cleanly outside hosted mode (stdio / SQLite / Redis-only)."""
    if not os.environ.get("MAGNET_DATABASE_URL"):
        return None
    try:
        return get_pool()
    except Exception as e:
        logger.debug(f"[postgres] pool unavailable: {e}")
        return None


def run_migrations(pool: Any, migrations_dir: str | Path | None = None) -> None:
    """Run every *.sql file in migrations_dir, sorted by filename. Every
    statement in those files is CREATE ... IF NOT EXISTS, so this is safe
    to call on every http_server.py startup."""
    if migrations_dir is None:
        env_dir = os.environ.get("MAGNET_MIGRATIONS_DIR")
        migrations_dir = Path(env_dir) if env_dir else Path(__file__).resolve().parents[2] / "migrations"
    migrations_dir = Path(migrations_dir)
    if not migrations_dir.exists():
        logger.warning(f"[postgres] migrations dir not found: {migrations_dir}")
        return
    sql_files = sorted(migrations_dir.glob("*.sql"))
    with pool.connection() as conn:
        for f in sql_files:
            conn.execute(f.read_text(encoding="utf-8"))
    logger.info(f"[postgres] ran {len(sql_files)} migration file(s) from {migrations_dir}")


def _glob_to_like(pattern: str) -> str:
    """Translate a `*`-wildcard glob (the only wildcard SQLiteBackend's
    fnmatch usage relies on) into a SQL LIKE pattern."""
    escaped = pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped.replace("*", "%")


class PostgresBackend:
    """Minimal Redis-compatible interface backed by Postgres."""

    def __init__(self, database_url: str | None = None):
        self._pool = get_pool(database_url)
        self._init_schema()

    def _init_schema(self) -> None:
        # Defensive/idempotent — the 4 generic tables only. Standalone-usable
        # even if migrations/0001_init.sql was never run separately.
        with self._pool.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kv (
                    key         TEXT PRIMARY KEY,
                    value       TEXT NOT NULL,
                    expires_at  DOUBLE PRECISION
                );
                CREATE TABLE IF NOT EXISTS lists (
                    id          BIGSERIAL PRIMARY KEY,
                    key         TEXT NOT NULL,
                    value       TEXT NOT NULL,
                    position    DOUBLE PRECISION NOT NULL DEFAULT 0,
                    expires_at  DOUBLE PRECISION
                );
                CREATE INDEX IF NOT EXISTS idx_lists_key ON lists(key, position);
                CREATE TABLE IF NOT EXISTS zsets (
                    key         TEXT NOT NULL,
                    member      TEXT NOT NULL,
                    score       DOUBLE PRECISION NOT NULL,
                    expires_at  DOUBLE PRECISION,
                    PRIMARY KEY (key, member)
                );
                CREATE INDEX IF NOT EXISTS idx_zsets_score ON zsets(key, score);
                CREATE TABLE IF NOT EXISTS hashes (
                    key    TEXT NOT NULL,
                    field  TEXT NOT NULL,
                    value  TEXT NOT NULL,
                    expires_at DOUBLE PRECISION,
                    PRIMARY KEY (key, field)
                );
                ALTER TABLE hashes ADD COLUMN IF NOT EXISTS expires_at DOUBLE PRECISION;
            """)

    def ping(self) -> bool:
        with self._pool.connection() as conn:
            conn.execute("SELECT 1")
        return True

    # ── String ops ─────────────────────────────────────────────────────

    def get(self, key: str) -> str | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM kv WHERE key = %s", (key,)
            ).fetchone()
            if row is None:
                return None
            value, expires_at = row
            if expires_at is not None and time.time() > expires_at:
                conn.execute("DELETE FROM kv WHERE key = %s", (key,))
                return None
            return value

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        expires_at = (time.time() + ex) if ex else None
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO kv (key, value, expires_at) VALUES (%s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET value = excluded.value, expires_at = excluded.expires_at
                """,
                (key, value, expires_at),
            )

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.set(key, value, ex=ttl)

    def delete(self, *keys: str) -> int:
        count = 0
        with self._pool.connection() as conn:
            for key in keys:
                count += conn.execute("DELETE FROM kv WHERE key = %s", (key,)).rowcount
                count += conn.execute("DELETE FROM lists WHERE key = %s", (key,)).rowcount
                count += conn.execute("DELETE FROM zsets WHERE key = %s", (key,)).rowcount
                count += conn.execute("DELETE FROM hashes WHERE key = %s", (key,)).rowcount
        return count

    def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if self.get(k) is not None)

    def expire(self, key: str, seconds: int) -> bool:
        expires_at = time.time() + seconds
        with self._pool.connection() as conn:
            conn.execute("UPDATE kv SET expires_at = %s WHERE key = %s", (expires_at, key))
            conn.execute("UPDATE lists SET expires_at = %s WHERE key = %s", (expires_at, key))
            conn.execute("UPDATE zsets SET expires_at = %s WHERE key = %s", (expires_at, key))
            conn.execute("UPDATE hashes SET expires_at = %s WHERE key = %s", (expires_at, key))
        return True

    def incr(self, key: str) -> int:
        """Increment a string counter, resetting an expired value."""
        with self._pool.connection() as conn:
            conn.execute(
                "DELETE FROM kv WHERE key = %s AND expires_at IS NOT NULL AND expires_at <= %s",
                (key, time.time()),
            )
            row = conn.execute(
                """
                INSERT INTO kv (key, value, expires_at) VALUES (%s, '1', NULL)
                ON CONFLICT (key) DO UPDATE SET value = (kv.value::bigint + 1)::text
                RETURNING value::bigint
                """,
                (key,),
            ).fetchone()
        return int(row[0])

    # ── List ops ───────────────────────────────────────────────────────

    def rpush(self, key: str, *values: str) -> int:
        with self._pool.connection() as conn:
            self._purge_expired_collections(conn, key)
            max_pos = conn.execute(
                "SELECT COALESCE(MAX(position), -1) FROM lists WHERE key = %s", (key,)
            ).fetchone()[0]
            for i, value in enumerate(values):
                conn.execute(
                    "INSERT INTO lists (key, value, position) VALUES (%s, %s, %s)",
                    (key, value, max_pos + 1 + i),
                )
            return conn.execute("SELECT COUNT(*) FROM lists WHERE key = %s", (key,)).fetchone()[0]

    def lpush(self, key: str, *values: str) -> int:
        with self._pool.connection() as conn:
            self._purge_expired_collections(conn, key)
            min_pos = conn.execute(
                "SELECT COALESCE(MIN(position), 0) FROM lists WHERE key = %s", (key,)
            ).fetchone()[0]
            for i, value in enumerate(values):
                conn.execute(
                    "INSERT INTO lists (key, value, position) VALUES (%s, %s, %s)",
                    (key, value, min_pos - len(values) + i),
                )
            return conn.execute("SELECT COUNT(*) FROM lists WHERE key = %s", (key,)).fetchone()[0]

    def llen(self, key: str) -> int:
        with self._pool.connection() as conn:
            self._purge_expired_collections(conn, key)
            return conn.execute("SELECT COUNT(*) FROM lists WHERE key = %s", (key,)).fetchone()[0]

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        with self._pool.connection() as conn:
            self._purge_expired_collections(conn, key)
            rows = conn.execute(
                "SELECT value FROM lists WHERE key = %s ORDER BY position", (key,)
            ).fetchall()
        values = [r[0] for r in rows]
        if end == -1:
            return values[start:]
        return values[start : end + 1]

    def ltrim(self, key: str, start: int, end: int) -> None:
        with self._pool.connection() as conn:
            self._purge_expired_collections(conn, key)
            rows = conn.execute(
                "SELECT id FROM lists WHERE key = %s ORDER BY position", (key,)
            ).fetchall()
            all_ids = [r[0] for r in rows]
            if end == -1:
                keep = set(all_ids[start:])
            else:
                keep = set(all_ids[start : end + 1])
            drop = [i for i in all_ids if i not in keep]
            if drop:
                conn.execute(
                    f"DELETE FROM lists WHERE id IN ({','.join(['%s'] * len(drop))})", drop
                )

    def eval(self, script: str, numkeys: int, *keys_and_args: str) -> list:  # noqa: ARG002
        """Narrow support for SignalBuffer's one atomic 'LRANGE 0 -1 then
        DEL' script only — not a general Lua interpreter."""
        key = keys_and_args[0]
        with self._pool.connection() as conn:
            self._purge_expired_collections(conn, key)
            rows = conn.execute(
                "SELECT value FROM lists WHERE key = %s ORDER BY position", (key,)
            ).fetchall()
            values = [r[0] for r in rows]
            conn.execute("DELETE FROM lists WHERE key = %s", (key,))
        return values

    def pipeline(self) -> "_PgPipeline":
        return _PgPipeline(self)

    # ── Sorted set ops ─────────────────────────────────────────────────

    def zadd(self, key: str, mapping: dict) -> int:
        with self._pool.connection() as conn:
            self._purge_expired_collections(conn, key)
            for member, score in mapping.items():
                conn.execute(
                    """
                    INSERT INTO zsets (key, member, score) VALUES (%s, %s, %s)
                    ON CONFLICT (key, member) DO UPDATE SET score = excluded.score
                    """,
                    (key, member, float(score)),
                )
        return len(mapping)

    def zrevrange(self, key: str, start: int, end: int) -> list[str]:
        with self._pool.connection() as conn:
            self._purge_expired_collections(conn, key)
            rows = conn.execute(
                "SELECT member FROM zsets WHERE key = %s ORDER BY score DESC", (key,)
            ).fetchall()
        members = [r[0] for r in rows]
        if end == -1:
            return members[start:]
        return members[start : end + 1]

    def zrange(self, key: str, start: int, end: int) -> list[str]:
        with self._pool.connection() as conn:
            self._purge_expired_collections(conn, key)
            rows = conn.execute(
                "SELECT member FROM zsets WHERE key = %s ORDER BY score ASC", (key,)
            ).fetchall()
        members = [r[0] for r in rows]
        if end == -1:
            return members[start:]
        return members[start : end + 1]

    def zremrangebyrank(self, key: str, start: int, end: int) -> int:
        with self._pool.connection() as conn:
            self._purge_expired_collections(conn, key)
            rows = conn.execute(
                "SELECT member FROM zsets WHERE key = %s ORDER BY score ASC", (key,)
            ).fetchall()
            members = [r[0] for r in rows]
            n = len(members)
            lo = start if start >= 0 else max(0, n + start)
            hi = end if end >= 0 else n + end
            to_delete = members[lo : hi + 1]
            if not to_delete:
                return 0
            conn.execute(
                f"DELETE FROM zsets WHERE key = %s AND member IN ({','.join(['%s'] * len(to_delete))})",
                [key] + to_delete,
            )
        return len(to_delete)

    def zcard(self, key: str) -> int:
        with self._pool.connection() as conn:
            self._purge_expired_collections(conn, key)
            return conn.execute("SELECT COUNT(*) FROM zsets WHERE key = %s", (key,)).fetchone()[0]

    # ── Hash ops ─────────────────────────────────────────────────────────

    def hset(self, key: str, field: str, value: str) -> None:
        with self._pool.connection() as conn:
            self._purge_expired_hash(conn, key)
            conn.execute(
                """
                INSERT INTO hashes (key, field, value, expires_at) VALUES (%s, %s, %s, NULL)
                ON CONFLICT (key, field) DO UPDATE SET value = excluded.value
                """,
                (key, field, value),
            )

    def hgetall(self, key: str) -> dict:
        with self._pool.connection() as conn:
            self._purge_expired_hash(conn, key)
            rows = conn.execute("SELECT field, value FROM hashes WHERE key = %s", (key,)).fetchall()
        return {field: value for field, value in rows}

    def hincrby(self, key: str, field: str, amount: int) -> int:
        with self._pool.connection() as conn:
            self._purge_expired_hash(conn, key)
            row = conn.execute(
                """
                INSERT INTO hashes (key, field, value, expires_at) VALUES (%s, %s, %s, NULL)
                ON CONFLICT (key, field) DO UPDATE SET value = (hashes.value::bigint + %s)::text
                RETURNING value::bigint
                """,
                (key, field, str(amount), amount),
            ).fetchone()
        return int(row[0])

    # ── Scan ─────────────────────────────────────────────────────────────

    def scan_iter(self, pattern: str):
        """Yield all kv keys matching a glob pattern (* wildcard only)."""
        like = _glob_to_like(pattern)
        now = time.time()
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT key, expires_at FROM kv WHERE key LIKE %s ESCAPE '\\'", (like,)
            ).fetchall()
        for key, expires_at in rows:
            if expires_at is not None and now > expires_at:
                continue
            yield key

    @staticmethod
    def _purge_expired_collections(conn: Any, key: str) -> None:
        now = time.time()
        conn.execute(
            "DELETE FROM lists WHERE key = %s AND expires_at IS NOT NULL AND expires_at <= %s",
            (key, now),
        )
        conn.execute(
            "DELETE FROM zsets WHERE key = %s AND expires_at IS NOT NULL AND expires_at <= %s",
            (key, now),
        )

    @staticmethod
    def _purge_expired_hash(conn: Any, key: str) -> None:
        conn.execute(
            "DELETE FROM hashes WHERE key = %s AND expires_at IS NOT NULL AND expires_at <= %s",
            (key, time.time()),
        )


class _PgPipeline:
    """Sequential command batch — mirrors local_store._Pipeline's API."""

    def __init__(self, backend: PostgresBackend):
        self._b = backend
        self._ops: list[tuple] = []

    def rpush(self, key: str, value: str) -> "_PgPipeline":
        self._ops.append(("rpush", key, value))
        return self

    def lpush(self, key: str, value: str) -> "_PgPipeline":
        self._ops.append(("lpush", key, value))
        return self

    def expire(self, key: str, seconds: int) -> "_PgPipeline":
        self._ops.append(("expire", key, seconds))
        return self

    def ltrim(self, key: str, start: int, end: int) -> "_PgPipeline":
        self._ops.append(("ltrim", key, start, end))
        return self

    def setex(self, key: str, ttl: int, value: str) -> "_PgPipeline":
        self._ops.append(("setex", key, ttl, value))
        return self

    def incr(self, key: str) -> "_PgPipeline":
        self._ops.append(("incr", key))
        return self

    def delete(self, *keys: str) -> "_PgPipeline":
        self._ops.append(("delete", *keys))
        return self

    def hset(self, key: str, field: str, value: str) -> "_PgPipeline":
        self._ops.append(("hset", key, field, value))
        return self

    def execute(self) -> list:
        results = [getattr(self._b, name)(*args) for name, *args in self._ops]
        self._ops.clear()
        return results
