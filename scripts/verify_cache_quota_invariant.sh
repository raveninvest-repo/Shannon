#!/usr/bin/env bash
# Verifies that cache_aware_total_tokens accurately reflects all token classes
# for every row written under or backfilled by migration 121.
#
# Invariant (per row):
#   cache_aware_total_tokens
#     == prompt_tokens + completion_tokens + cache_read_tokens + cache_creation_tokens
#
# Reports:
#   - Per-row mismatch counts (token_usage, task_executions)
#   - Cache leak recovered (true_total - old_total over token_usage)
#
# Environments:
#   - Local docker compose (default):
#       ./scripts/verify_cache_quota_invariant.sh
#   - EKS / k8s with kubectl-run psql pod:
#       PSQL_CMD="kubectl exec -i postgres-client -- psql -U shannon -d shannon -t -A" \
#         ./scripts/verify_cache_quota_invariant.sh
#   - Any direct psql client (RDS, port-forward, etc.):
#       PSQL_CMD="psql postgresql://user:pass@host/shannon -t -A" \
#         ./scripts/verify_cache_quota_invariant.sh
#
set -euo pipefail

# PSQL_CMD lets the caller swap the docker-compose default for an EKS/RDS
# kubectl invocation. We use `read -ra` to keep flags as separate argv tokens.
DEFAULT_PSQL="docker exec shannon-postgres-1 psql -U shannon -d shannon -t -A"
read -ra PSQL <<< "${PSQL_CMD:-$DEFAULT_PSQL}"

echo "=== Per-row invariant check (token_usage) ==="
"${PSQL[@]}" -c "
SELECT COUNT(*) AS mismatch_rows
  FROM token_usage
 WHERE cache_aware_total_tokens != prompt_tokens + completion_tokens
                                  + cache_read_tokens + cache_creation_tokens;"

echo "=== Per-task invariant check (task_executions, COMPLETED only) ==="
"${PSQL[@]}" -c "
SELECT COUNT(*) AS mismatch_rows
  FROM task_executions
 WHERE cache_aware_total_tokens != prompt_tokens + completion_tokens
                                  + cache_read_tokens + cache_creation_tokens
   AND status = 'COMPLETED';"

echo "=== Cache leak measurement (token_usage SUMs) ==="
"${PSQL[@]}" -c "
SELECT
  SUM(cache_aware_total_tokens) AS true_total,
  SUM(total_tokens)             AS old_total,
  SUM(cache_aware_total_tokens) - SUM(total_tokens) AS leak_recovered,
  SUM(cache_read_tokens)        AS sum_cache_read,
  SUM(cache_creation_tokens)    AS sum_cache_creation
FROM token_usage;"
