-- Migration 121: Add cache-aware total token tracking columns
-- Adds parallel cache_aware_total_tokens = prompt + completion + cache_read + cache_creation
-- to support quota tracking that includes prompt cache cost without changing the
-- existing total_tokens semantics (kept as input + output for OpenAI compatibility).

-- token_usage table (per-LLM-call ledger)
ALTER TABLE token_usage
    ADD COLUMN IF NOT EXISTS cache_aware_total_tokens INTEGER DEFAULT 0;

-- task_executions table (per-workflow rollup)
ALTER TABLE task_executions
    ADD COLUMN IF NOT EXISTS cache_aware_total_tokens INTEGER DEFAULT 0;

-- Backfill existing rows so historical quota queries are correct.
-- This is safe because cache_read_tokens and cache_creation_tokens
-- already exist (added by migration 112) and have NOT NULL defaults of 0.
--
-- Operational note (production / shannon-cloud):
-- The UPDATE below is a single statement and will hold a row-level write lock
-- on every matching row of token_usage / task_executions until commit. On
-- multi-million-row tables this can spike WAL traffic and block concurrent
-- writers. Run during a low-traffic window, or batch by id range when
-- applying to large clusters, e.g.:
--
--   UPDATE token_usage SET cache_aware_total_tokens = ...
--    WHERE cache_aware_total_tokens = 0
--      AND id BETWEEN $low AND $high;
--
-- The `WHERE cache_aware_total_tokens = 0` predicate makes both forms
-- idempotent.
UPDATE token_usage
   SET cache_aware_total_tokens = COALESCE(prompt_tokens, 0)
                                + COALESCE(completion_tokens, 0)
                                + COALESCE(cache_read_tokens, 0)
                                + COALESCE(cache_creation_tokens, 0)
 WHERE cache_aware_total_tokens = 0;

UPDATE task_executions
   SET cache_aware_total_tokens = COALESCE(prompt_tokens, 0)
                                + COALESCE(completion_tokens, 0)
                                + COALESCE(cache_read_tokens, 0)
                                + COALESCE(cache_creation_tokens, 0)
 WHERE cache_aware_total_tokens = 0;

-- This index is intentionally shaped for "top-N consumers" admin queries
-- (e.g. `ORDER BY cache_aware_total_tokens DESC LIMIT N`), NOT for the
-- per-session SUM aggregations in handlers/session.go. Those queries are
-- already covered by idx_task_user_session(user_id, session_id) and
-- idx_task_executions_tenant_user_created (migration 118). Keep this
-- partial DESC index for billing dashboards / outlier reports.
CREATE INDEX IF NOT EXISTS idx_task_executions_cache_aware_total
    ON task_executions (cache_aware_total_tokens DESC)
    WHERE cache_aware_total_tokens > 0;
