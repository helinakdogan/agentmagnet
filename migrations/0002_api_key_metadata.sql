-- Agent Magnet — API key metadata for dashboard key management
-- Idempotent: safe to run on every http_server.py startup.

ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS name TEXT NOT NULL DEFAULT '';
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS masked_key TEXT NOT NULL DEFAULT '';
