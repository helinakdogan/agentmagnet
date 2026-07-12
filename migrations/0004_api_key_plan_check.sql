-- Agent Magnet — constrain api_keys.plan to known values
-- api_keys.plan already existed (0001_init.sql, default 'free'); this just
-- adds a CHECK so a typo'd plan value can never silently defeat
-- team_permissions.py's PAID_PLANS gate. Idempotent: Postgres has no native
-- ADD CONSTRAINT IF NOT EXISTS, so this guards with a catalog lookup.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'api_keys_plan_check'
    ) THEN
        ALTER TABLE api_keys
            ADD CONSTRAINT api_keys_plan_check CHECK (plan IN ('free', 'team', 'pro'));
    END IF;
END $$;
