package budget

import (
	"context"
	"database/sql"
	"regexp"
	"testing"
	"time"

	sqlmock "github.com/DATA-DOG/go-sqlmock"
	"go.uber.org/zap"
)

func TestCheckBudget_DefaultsAllowSmallEstimate(t *testing.T) {
	bm := NewBudgetManager(&sql.DB{}, zap.NewNop())
	ctx := context.Background()
	res, err := bm.CheckBudget(ctx, "u1", "s1", "t1", 1000)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.CanProceed {
		t.Fatalf("expected CanProceed=true, got false: %+v", res)
	}
	if res.RemainingTaskBudget <= 0 || res.RemainingSessionBudget <= 0 {
		t.Fatalf("expected positive remaining budgets, got %+v", res)
	}
}

func TestEstimateCost_ModelPricing(t *testing.T) {
	bm := NewBudgetManager(&sql.DB{}, zap.NewNop())
	cost := bm.estimateCost(1000, "gpt-5-nano-2025-08-07")
	if cost <= 0 {
		t.Fatalf("expected positive cost for 1k tokens, got %f", cost)
	}
}

func TestRecordUsage_ExecInsertsTokenUsage(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	bm := NewBudgetManager(db, zap.NewNop())
	usage := &BudgetTokenUsage{
		UserID: "u1", SessionID: "s1", TaskID: "t1", AgentID: "a1",
		Model: "gpt-5-nano-2025-08-07", Provider: "openai", InputTokens: 10, OutputTokens: 20,
	}

	// Expect user lookup
	mock.ExpectQuery("SELECT id FROM users WHERE external_id").
		WithArgs("u1").
		WillReturnError(sql.ErrNoRows)

	// Expect user creation
	userID := "12345678-1234-5678-1234-567812345678"
	mock.ExpectQuery("INSERT INTO users").
		WithArgs(sqlmock.AnyArg(), "u1").
		WillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow(userID))

	// Expect task lookup
	taskID := "87654321-4321-8765-4321-876543218765"
	mock.ExpectQuery("SELECT id FROM task_executions WHERE workflow_id").
		WithArgs("t1").
		WillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow(taskID))

	mock.ExpectExec(regexp.QuoteMeta(
		"INSERT INTO token_usage",
	)).WithArgs(
		sqlmock.AnyArg(), sqlmock.AnyArg(), usage.AgentID, usage.Provider, usage.Model,
		sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(),
		sqlmock.AnyArg(), sqlmock.AnyArg(), // cache_read_tokens, cache_creation_tokens
		sqlmock.AnyArg(),                   // cache_aware_total_tokens
		sqlmock.AnyArg(),                   // call_sequence
	).WillReturnResult(sqlmock.NewResult(1, 1))

	if err := bm.RecordUsage(context.Background(), usage); err != nil {
		t.Fatalf("RecordUsage error: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestGetUsageReport_AggregatesRows(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	bm := NewBudgetManager(db, zap.NewNop())

	rows := sqlmock.NewRows([]string{"user_id", "task_id", "model", "provider", "input_total", "output_total", "total_tokens", "cache_aware_total_tokens", "total_cost", "request_count"}).
		AddRow("u1", "t1", "gpt-5-nano-2025-08-07", "openai", 30, 60, 90, 90, 0.1, 2)

	mock.ExpectQuery(`SELECT\s+tu\.user_id,.*FROM\s+token_usage`).
		WithArgs(sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg()).
		WillReturnRows(rows)

	from := time.Now().Add(-time.Hour)
	to := time.Now()
	rep, err := bm.GetUsageReport(context.Background(), UsageFilters{StartTime: from, EndTime: to})
	if err != nil {
		t.Fatalf("GetUsageReport error: %v", err)
	}
	if rep.TotalTokens != 90 || rep.TotalCostUSD <= 0 {
		t.Fatalf("unexpected report: %+v", rep)
	}
}

func TestRecordUsage_CostOverrideSkipsPricing(t *testing.T) {
	// When CostOverride > 0, RecordUsage should use it instead of pricing calculation
	bm := NewBudgetManager(nil, zap.NewNop()) // nil db = skip persistence

	// Set up a session budget so RecordUsage updates in-memory state
	bm.SetSessionBudget("s1", &TokenBudget{
		TaskBudget:    100000,
		SessionBudget: 100000,
	})

	overrideCost := 0.0099 // Python-reported real cost
	usage := &BudgetTokenUsage{
		UserID:       "u1",
		SessionID:    "s1",
		TaskID:       "t1",
		AgentID:      "tool_web_fetch",
		Model:        "shannon_web_fetch",
		Provider:     "shannon-scraper",
		InputTokens:  0,
		OutputTokens: 27000, // synthetic tokens that would price high without override
		CostOverride: overrideCost,
	}

	if err := bm.RecordUsage(context.Background(), usage); err != nil {
		t.Fatalf("RecordUsage error: %v", err)
	}

	// CostUSD should be the override value, not the pricing calculation
	if usage.CostUSD != overrideCost {
		t.Fatalf("expected CostUSD=%f (from CostOverride), got %f", overrideCost, usage.CostUSD)
	}
}

func TestRecordUsage_NoCostOverrideFallsToPricing(t *testing.T) {
	// When CostOverride is 0, RecordUsage should use pricing calculation as before
	bm := NewBudgetManager(nil, zap.NewNop())

	bm.SetSessionBudget("s1", &TokenBudget{
		TaskBudget:    100000,
		SessionBudget: 100000,
	})

	usage := &BudgetTokenUsage{
		UserID:       "u1",
		SessionID:    "s1",
		TaskID:       "t1",
		AgentID:      "tool_web_search",
		Model:        "shannon_web_search",
		Provider:     "shannon-scraper",
		InputTokens:  0,
		OutputTokens: 7500,
		CostOverride: 0, // No override — should use pricing
	}

	if err := bm.RecordUsage(context.Background(), usage); err != nil {
		t.Fatalf("RecordUsage error: %v", err)
	}

	// CostUSD should be calculated via pricing, not zero
	if usage.CostUSD <= 0 {
		t.Fatalf("expected CostUSD > 0 from pricing calculation, got %f", usage.CostUSD)
	}
}

// CheckBudget reads sessionBudget.SessionTokensUsed; that counter must
// be incremented by the cache-aware total so cache-heavy sessions
// don't silently overrun their quota.
func TestRecordUsage_BudgetCounterUsesCacheAware(t *testing.T) {
	bm := NewBudgetManager(nil, zap.NewNop())
	bm.SetSessionBudget("sess-budget", &TokenBudget{
		TaskBudget:    100000,
		SessionBudget: 100000,
	})

	usage := &BudgetTokenUsage{
		UserID:              "u1",
		SessionID:           "sess-budget",
		Model:               "claude-sonnet-4-5-20250929",
		Provider:            "anthropic",
		InputTokens:         100,
		OutputTokens:        50,
		CacheReadTokens:     500,
		CacheCreationTokens: 200,
	}
	if err := bm.RecordUsage(context.Background(), usage); err != nil {
		t.Fatalf("RecordUsage error: %v", err)
	}

	bm.mu.RLock()
	sb := bm.sessionBudgets["sess-budget"]
	bm.mu.RUnlock()

	// Quota counter must include cache classes (= 100 + 50 + 500 + 200 = 850)
	if got, want := sb.SessionTokensUsed, 850; got != want {
		t.Errorf("SessionTokensUsed = %d, want %d (cache-aware total)", got, want)
	}
	if got, want := sb.TaskTokensUsed, 850; got != want {
		t.Errorf("TaskTokensUsed = %d, want %d (cache-aware total)", got, want)
	}
}

func TestRecordUsage_CacheAwareTotalInvariant(t *testing.T) {
	bm := NewBudgetManager(nil, zap.NewNop())
	bm.SetSessionBudget("sess-cache", &TokenBudget{
		TaskBudget:    100000,
		SessionBudget: 100000,
	})

	usage := &BudgetTokenUsage{
		UserID:              "u1",
		SessionID:           "sess-cache",
		Model:               "claude-sonnet-4-5-20250929",
		Provider:            "anthropic",
		InputTokens:         100,
		OutputTokens:        50,
		CacheReadTokens:     500,
		CacheCreationTokens: 200,
	}
	if err := bm.RecordUsage(context.Background(), usage); err != nil {
		t.Fatalf("RecordUsage error: %v", err)
	}

	// total_tokens stays input + output (OpenAI compatible)
	if got, want := usage.TotalTokens, 150; got != want {
		t.Errorf("TotalTokens = %d, want %d (input+output, OpenAI compat)", got, want)
	}
	// cache_aware_total_tokens adds cache classes
	if got, want := usage.CacheAwareTotalTokens, 850; got != want {
		t.Errorf("CacheAwareTotalTokens = %d, want %d (input+output+cache_read+cache_creation)", got, want)
	}
}

func TestRecordUsage_1hCacheCreationPricedCorrectly(t *testing.T) {
	bm := NewBudgetManager(nil, zap.NewNop())

	bm.SetSessionBudget("sess1", &TokenBudget{
		TaskBudget:    100000,
		SessionBudget: 100000,
	})

	usage := &BudgetTokenUsage{
		UserID:                "u1",
		SessionID:             "sess1",
		Model:                 "claude-sonnet-4-5-20250929",
		Provider:              "anthropic",
		InputTokens:           1000,
		OutputTokens:          500,
		CacheCreationTokens:   3000,
		CacheCreation1hTokens: 2000,
	}
	if err := bm.RecordUsage(context.Background(), usage); err != nil {
		t.Fatalf("RecordUsage error: %v", err)
	}

	// claude-sonnet-4-5-20250929: input 0.003/1K, output 0.015/1K
	// base = 1000/1000 * 0.003 + 500/1000 * 0.015 = 0.0105
	// 5m: 1000/1000 * 0.003 * 1.25 = 0.00375
	// 1h: 2000/1000 * 0.003 * 2.0  = 0.012
	// expected = 0.0105 + 0.00375 + 0.012 = 0.02625
	if usage.CostUSD < 0.026 || usage.CostUSD > 0.027 {
		t.Errorf("CostUSD = %f, want ~0.02625", usage.CostUSD)
	}
}
