-- Agent Magnet — index usage_events for team-scoped queries
-- 0001_init.sql only indexed (user_id, created_at). team_permissions.py's
-- get_team_sync_usage() filters by (team_id, created_at) — without this,
-- every monthly-sync-cap check (now run on every check_team_permission
-- call) would be a full table scan on a table that grows on every team
-- read/write, forever. Idempotent.

CREATE INDEX IF NOT EXISTS idx_usage_events_team_time ON usage_events(team_id, created_at);
