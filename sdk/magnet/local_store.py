"""
SQLiteBackend
-------------
Redis-compatible storage backend using a local SQLite database.

Activated when MAGNET_LOCAL_MODE=1 or when no MAGNET_REDIS_URL is provided
and local mode is chosen during `agent-magnet init`.

Implements the subset of the Redis API used by:
  - ProfileStore  (kv: setex, get, delete, lpush, ltrim)
  - SignalBuffer  (lists: pipeline/rpush/expire, llen, lrange, eval)
  - EpisodicStore (sorted sets: zadd, zrevrange, zremrangebyrank, expire)
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / ".agent-magnet" / "memory.db"


class SQLiteBackend:
    """Minimal Redis-compatible interface backed by a local SQLite file."""

    def __init__(self, db_path: str | Path | None = None):
        self._path = Path(db_path or DEFAULT_DB_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL
                );
                CREATE TABLE IF NOT EXISTS lists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    position REAL NOT NULL DEFAULT 0,
                    expires_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_lists_key ON lists(key, position);
                CREATE TABLE IF NOT EXISTS zsets (
                    key TEXT NOT NULL,
                    member TEXT NOT NULL,
                    score REAL NOT NULL,
                    expires_at REAL,
                    PRIMARY KEY (key, member)
                );
                CREATE INDEX IF NOT EXISTS idx_zsets_score ON zsets(key, score);
            """)

    def ping(self) -> bool:
        return True

    def _purge_expired_collections(self, key: str) -> None:
        """Remove expired list and sorted-set rows before collection access."""
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM lists WHERE key = ? AND expires_at IS NOT NULL AND expires_at <= ?",
                (key, now),
            )
            self._conn.execute(
                "DELETE FROM zsets WHERE key = ? AND expires_at IS NOT NULL AND expires_at <= ?",
                (key, now),
            )

    # ── String ops ─────────────────────────────────────────────────────

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM kv WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        value, expires_at = row
        if expires_at is not None and time.time() > expires_at:
            with self._lock, self._conn:
                self._conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            return None
        return value

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        expires_at = (time.time() + ex) if ex else None
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value, expires_at) VALUES (?, ?, ?)",
                (key, value, expires_at),
            )

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.set(key, value, ex=ttl)

    def delete(self, *keys: str) -> int:
        count = 0
        with self._lock, self._conn:
            for key in keys:
                count += self._conn.execute("DELETE FROM kv WHERE key = ?", (key,)).rowcount
                count += self._conn.execute(
                    "DELETE FROM kv WHERE key = ?", (f"__hash__{key}",)
                ).rowcount
                count += self._conn.execute("DELETE FROM lists WHERE key = ?", (key,)).rowcount
                count += self._conn.execute("DELETE FROM zsets WHERE key = ?", (key,)).rowcount
        return count

    def incr(self, key: str) -> int:
        now = time.time()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT value, expires_at FROM kv WHERE key = ?", (key,)
            ).fetchone()
            if row is None or (row[1] is not None and row[1] <= now):
                new_value, expires_at = 1, None
            else:
                new_value, expires_at = int(row[0]) + 1, row[1]
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value, expires_at) VALUES (?, ?, ?)",
                (key, str(new_value), expires_at),
            )
        return new_value

    def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if self.get(k) is not None)

    def expire(self, key: str, seconds: int) -> bool:
        expires_at = time.time() + seconds
        with self._lock, self._conn:
            self._conn.execute("UPDATE kv SET expires_at = ? WHERE key = ?", (expires_at, key))
            self._conn.execute(
                "UPDATE kv SET expires_at = ? WHERE key = ?",
                (expires_at, f"__hash__{key}"),
            )
            self._conn.execute("UPDATE lists SET expires_at = ? WHERE key = ?", (expires_at, key))
            self._conn.execute("UPDATE zsets SET expires_at = ? WHERE key = ?", (expires_at, key))
        return True

    # ── List ops ───────────────────────────────────────────────────────

    def rpush(self, key: str, *values: str) -> int:
        self._purge_expired_collections(key)
        with self._lock:
            with self._conn:
                max_pos = self._conn.execute(
                    "SELECT COALESCE(MAX(position), -1) FROM lists WHERE key = ?", (key,)
                ).fetchone()[0]
                for i, value in enumerate(values):
                    self._conn.execute(
                        "INSERT INTO lists (key, value, position) VALUES (?, ?, ?)",
                        (key, value, max_pos + 1 + i),
                    )
            return self._conn.execute(
                "SELECT COUNT(*) FROM lists WHERE key = ?", (key,)
            ).fetchone()[0]

    def lpush(self, key: str, *values: str) -> int:
        self._purge_expired_collections(key)
        with self._lock:
            with self._conn:
                min_pos = self._conn.execute(
                    "SELECT COALESCE(MIN(position), 0) FROM lists WHERE key = ?", (key,)
                ).fetchone()[0]
                for i, value in enumerate(values):
                    self._conn.execute(
                        "INSERT INTO lists (key, value, position) VALUES (?, ?, ?)",
                        (key, value, min_pos - len(values) + i),
                    )
            return self._conn.execute(
                "SELECT COUNT(*) FROM lists WHERE key = ?", (key,)
            ).fetchone()[0]

    def llen(self, key: str) -> int:
        self._purge_expired_collections(key)
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM lists WHERE key = ?", (key,)
            ).fetchone()[0]

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        self._purge_expired_collections(key)
        with self._lock:
            rows = self._conn.execute(
                "SELECT value FROM lists WHERE key = ? ORDER BY position", (key,)
            ).fetchall()
        values = [r[0] for r in rows]
        if end == -1:
            return values[start:]
        return values[start : end + 1]

    def ltrim(self, key: str, start: int, end: int) -> None:
        self._purge_expired_collections(key)
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM lists WHERE key = ? ORDER BY position", (key,)
            ).fetchall()
        all_ids = [r[0] for r in rows]
        if end == -1:
            keep = set(all_ids[start:])
        else:
            keep = set(all_ids[start : end + 1])
        drop = [i for i in all_ids if i not in keep]
        if drop:
            with self._lock, self._conn:
                self._conn.execute(
                    f"DELETE FROM lists WHERE id IN ({','.join('?' * len(drop))})", drop
                )

    def eval(self, script: str, numkeys: int, *keys_and_args: str) -> list:
        """Handles SignalBuffer's atomic LRANGE + DEL flush."""
        key = keys_and_args[0]
        self._purge_expired_collections(key)
        with self._lock:
            rows = self._conn.execute(
                "SELECT value FROM lists WHERE key = ? ORDER BY position", (key,)
            ).fetchall()
            values = [r[0] for r in rows]
            with self._conn:
                self._conn.execute("DELETE FROM lists WHERE key = ?", (key,))
        return values

    def pipeline(self) -> "_Pipeline":
        return _Pipeline(self)

    # ── Sorted set ops ─────────────────────────────────────────────────

    def zadd(self, key: str, mapping: dict) -> int:
        self._purge_expired_collections(key)
        with self._lock, self._conn:
            for member, score in mapping.items():
                self._conn.execute(
                    "INSERT OR REPLACE INTO zsets (key, member, score) VALUES (?, ?, ?)",
                    (key, member, float(score)),
                )
        return len(mapping)

    def zrevrange(self, key: str, start: int, end: int) -> list[str]:
        self._purge_expired_collections(key)
        with self._lock:
            rows = self._conn.execute(
                "SELECT member FROM zsets WHERE key = ? ORDER BY score DESC", (key,)
            ).fetchall()
        members = [r[0] for r in rows]
        if end == -1:
            return members[start:]
        return members[start : end + 1]

    def zrange(self, key: str, start: int, end: int) -> list[str]:
        self._purge_expired_collections(key)
        with self._lock:
            rows = self._conn.execute(
                "SELECT member FROM zsets WHERE key = ? ORDER BY score ASC", (key,)
            ).fetchall()
        members = [r[0] for r in rows]
        if end == -1:
            return members[start:]
        return members[start : end + 1]

    def zremrangebyrank(self, key: str, start: int, end: int) -> int:
        self._purge_expired_collections(key)
        with self._lock:
            rows = self._conn.execute(
                "SELECT member FROM zsets WHERE key = ? ORDER BY score ASC", (key,)
            ).fetchall()
        members = [r[0] for r in rows]
        n = len(members)
        lo = start if start >= 0 else max(0, n + start)
        hi = end if end >= 0 else n + end
        to_delete = members[lo : hi + 1]
        if not to_delete:
            return 0
        with self._lock, self._conn:
            self._conn.execute(
                f"DELETE FROM zsets WHERE key = ? AND member IN ({','.join('?' * len(to_delete))})",
                [key] + to_delete,
            )
        return len(to_delete)

    def zcard(self, key: str) -> int:
        self._purge_expired_collections(key)
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM zsets WHERE key = ?", (key,)
            ).fetchone()[0]

    # ── Hash ops (used by profile index, usage counter) ────────────────

    def hset(self, key: str, field: str, value: str) -> None:
        import json as _json
        hkey = f"__hash__{key}"
        now = time.time()
        with self._lock, self._conn:
            raw = self._conn.execute(
                "SELECT value, expires_at FROM kv WHERE key = ?", (hkey,)
            ).fetchone()
            if raw and raw[1] is not None and raw[1] <= now:
                self._conn.execute("DELETE FROM kv WHERE key = ?", (hkey,))
                raw = None
            data = _json.loads(raw[0]) if raw else {}
            data[field] = value
            expires_at = raw[1] if raw else None
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value, expires_at) VALUES (?, ?, ?)",
                (hkey, _json.dumps(data), expires_at),
            )

    def hgetall(self, key: str) -> dict:
        import json as _json
        raw_value = self.get(f"__hash__{key}")
        if raw_value is None:
            return {}
        try:
            return _json.loads(raw_value)
        except Exception:
            return {}

    def hincrby(self, key: str, field: str, amount: int) -> int:
        import json as _json
        hkey = f"__hash__{key}"
        now = time.time()
        with self._lock, self._conn:
            raw = self._conn.execute(
                "SELECT value, expires_at FROM kv WHERE key = ?", (hkey,)
            ).fetchone()
            if raw and raw[1] is not None and raw[1] <= now:
                self._conn.execute("DELETE FROM kv WHERE key = ?", (hkey,))
                raw = None
            data = _json.loads(raw[0]) if raw else {}
            new_val = int(data.get(field, 0)) + amount
            data[field] = str(new_val)
            expires_at = raw[1] if raw else None
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value, expires_at) VALUES (?, ?, ?)",
                (hkey, _json.dumps(data), expires_at),
            )
        return new_val

    # ── Scan (used by ProjectStore scan, team scan) ─────────────────────

    def scan_iter(self, pattern: str):
        """Yield all kv keys matching a glob pattern (* wildcard only)."""
        import fnmatch
        import time as _time
        with self._lock:
            rows = self._conn.execute("SELECT key, expires_at FROM kv").fetchall()
        now = _time.time()
        for key, expires_at in rows:
            if expires_at is not None and now > expires_at:
                continue
            if key.startswith("__hash__"):
                continue  # skip internal hash entries
            if fnmatch.fnmatch(key, pattern):
                yield key


class _Pipeline:
    """Sequential command batch — mirrors the Redis pipeline API."""

    def __init__(self, backend: SQLiteBackend):
        self._b = backend
        self._ops: list[tuple] = []

    def rpush(self, key: str, value: str) -> "_Pipeline":
        self._ops.append(("rpush", key, value))
        return self

    def lpush(self, key: str, value: str) -> "_Pipeline":
        self._ops.append(("lpush", key, value))
        return self

    def expire(self, key: str, seconds: int) -> "_Pipeline":
        self._ops.append(("expire", key, seconds))
        return self

    def ltrim(self, key: str, start: int, end: int) -> "_Pipeline":
        self._ops.append(("ltrim", key, start, end))
        return self

    def setex(self, key: str, ttl: int, value: str) -> "_Pipeline":
        self._ops.append(("setex", key, ttl, value))
        return self

    def delete(self, *keys: str) -> "_Pipeline":
        self._ops.append(("delete", *keys))
        return self

    def hset(self, key: str, field: str, value: str) -> "_Pipeline":
        self._ops.append(("hset", key, field, value))
        return self

    def incr(self, key: str) -> "_Pipeline":
        self._ops.append(("incr", key))
        return self

    def execute(self) -> list:
        results = [getattr(self._b, name)(*args) for name, *args in self._ops]
        self._ops.clear()
        return results
