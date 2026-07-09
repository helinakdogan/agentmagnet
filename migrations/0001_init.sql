-- Agent Magnet — hosted Postgres schema
-- Idempotent: safe to run on every http_server.py startup (CREATE ... IF NOT EXISTS throughout).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- for gen_random_uuid()

-- ── Generic duck-typed backend tables (PostgresBackend / postgres_store.py) ──
-- Gives every existing store (MemoryStore, MagnetTeamStore, SignalBuffer,
-- EpisodicStore's redis-fallback path, UsageCounter's redis-hincrby path) a
-- drop-in Redis-shaped surface. Nothing above the storage layer changes.

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
    PRIMARY KEY (key, field)
);

-- ── Hosted relational tables ──────────────────────────────────────────────

-- LIVE — read by magnet.auth.validate_key() on every HTTP request.
CREATE TABLE IF NOT EXISTS api_keys (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash    TEXT NOT NULL UNIQUE,
    user_id     TEXT NOT NULL,
    team_id     TEXT,
    plan        TEXT NOT NULL DEFAULT 'free',
    active      BOOLEAN NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);

-- SCOPE CUT: not yet the live read/write path for recall/remember.
-- MemoryStore keeps writing to the generic kv table above (vmm:{user}:
-- {profile}:{project} JSON blobs) unchanged, even in hosted/Postgres mode.
-- Reserved for a future pgvector-backed semantic-search upgrade.
CREATE TABLE IF NOT EXISTS memory_items (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL,
    team_id     TEXT,
    profile     TEXT NOT NULL,
    project     TEXT NOT NULL,
    category    TEXT NOT NULL,
    text        TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.8,
    status      TEXT NOT NULL DEFAULT 'active',
    embedding   vector(384),  -- all-MiniLM-L6-v2 output dim (local_embeddings.py)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_memory_items_scope ON memory_items(user_id, profile, project);

-- SCOPE CUT: not yet wired into MagnetTeamStore. team_store.py keeps using
-- the generic kv/lists/zsets tables above unchanged (team:{id}:meta,
-- team:{id}:members, vmm:team:{id}:{project} keys). These two tables exist
-- to satisfy the schema spec but have no reader/writer yet.
CREATE TABLE IF NOT EXISTS teams (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    owner_user_id  TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS team_members (
    team_id  TEXT NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    user_id  TEXT NOT NULL,
    role     TEXT NOT NULL DEFAULT 'member',
    PRIMARY KEY (team_id, user_id)
);

-- LIVE — bill-ready metering log (accurate, always-on; see usage_counter.py
-- check_usage_limit for the currently-inert enforcement TODO seam).
-- Distinct from UsageCounter's older hincrby counters (magnet:usage:{user_id}
-- hash via the generic backend), which keep working unchanged for stdio mode.
CREATE TABLE IF NOT EXISTS usage_events (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    team_id     TEXT,
    event_type  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_usage_events_user_time ON usage_events(user_id, created_at);
