-- Agent Magnet — attribute usage_events to the specific key that made the call
-- Enables the dashboard's per-key usage breakdown (usage_counter.get_usage_by_key).
-- Nullable: historical rows and any future non-key-authenticated event source
-- have no key to attribute to. ON DELETE SET NULL since api_keys rows are
-- deactivated, never deleted — this is defensive, not expected to fire.
-- Idempotent.

ALTER TABLE usage_events ADD COLUMN IF NOT EXISTS key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_usage_events_key_time ON usage_events(key_id, created_at);
