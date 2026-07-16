-- Agent Magnet — append-only change log for team memory items (git-blame
-- style history). Written by team_store.py's _record_history() on every
-- team-item write (share_project, share_item, auto_promote_if_agreed);
-- read by the `history` MCP tool and the dashboard's Memory tab. Rows are
-- never updated or deleted by application code — append-only.
--
-- item_id/team_id are stored as plain text, not foreign keys: team items
-- live in Redis (vmm:team:{team_id}:{project}), not a Postgres table, so
-- there's nothing to reference here.

CREATE TABLE IF NOT EXISTS memory_history (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id    TEXT NOT NULL,
    team_id    TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    action     TEXT NOT NULL CHECK (action IN (
                   'created', 'edited', 'superseded', 'deleted',
                   'shared_to_team', 'promoted'
               )),
    old_text   TEXT,
    new_text   TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memory_history_team_item_time
    ON memory_history (team_id, item_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_history_team_time
    ON memory_history (team_id, created_at DESC);
