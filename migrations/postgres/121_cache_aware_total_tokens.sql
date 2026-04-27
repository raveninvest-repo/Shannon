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

-- Index for quota queries
CREATE INDEX IF NOT EXISTS idx_task_executions_cache_aware_total
    ON task_executions (cache_aware_total_tokens DESC)
    WHERE cache_aware_total_tokens > 0;
