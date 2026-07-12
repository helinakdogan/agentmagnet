-- Agent Magnet — team registry becomes the live permission authority
-- (teams/team_members existed as unused placeholders since 0001_init.sql;
-- this makes them the server-side source of truth for team coordination).
-- Idempotent: safe to run on every http_server.py startup.

ALTER TABLE teams ADD COLUMN IF NOT EXISTS storage_mode TEXT NOT NULL DEFAULT 'managed';
ALTER TABLE teams ADD COLUMN IF NOT EXISTS redis_url_enc TEXT;
ALTER TABLE teams ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'team';
ALTER TABLE teams ADD COLUMN IF NOT EXISTS sync_limit INTEGER NOT NULL DEFAULT 50000;
ALTER TABLE teams ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT true;

ALTER TABLE team_members ADD COLUMN IF NOT EXISTS joined_at TIMESTAMPTZ NOT NULL DEFAULT now();
