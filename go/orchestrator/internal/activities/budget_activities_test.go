package activities

import (
	"context"
	"testing"
	"time"

	"github.com/Kocoro-lab/Shannon/go/orchestrator/internal/budget"
	"go.uber.org/zap"
)

// Test that CheckTokenBudgetWithBackpressure returns appropriate delay value without blocking
func TestCheckTokenBudgetWithBackpressure_ReturnsDelayValue(t *testing.T) {
	mgr := budget.NewBudgetManager(nil, zap.NewNop())
	acts := NewBudgetActivitiesWithManager(mgr, zap.NewNop())

	userID := "u-delay"
	sessionID := "s-delay"

	// Configure session budget so a modest estimate crosses 80%
	mgr.SetSessionBudget(sessionID, &budget.TokenBudget{
		TaskBudget:        1000,
		SessionBudget:     1000,
		SessionTokensUsed: 700,
		HardLimit:         true,
	})

	// This request projects usage to exactly 80% (800/1000)
	in := BudgetCheckInput{
		UserID:          userID,
		SessionID:       sessionID,
		TaskID:          "task-delay",
		EstimatedTokens: 100,
	}

	start := time.Now()
	res, err := acts.CheckTokenBudgetWithBackpressure(context.Background(), in)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("activity error: %v", err)
	}
	if !res.BackpressureActive || res.BackpressureDelay <= 0 {
		t.Fatalf("expected backpressure with positive delay value, got: %+v", res)
	}

	// Activity should NOT block - it returns immediately with delay value for workflow to handle
	// This ensures Temporal workers are not blocked
	if elapsed > 50*time.Millisecond {
		t.Fatalf("activity should return immediately without blocking, but took %v", elapsed)
	}

	// Verify delay value is correct for 80% usage
	// Based on calculateBackpressureDelay: 80-85% returns 50ms
	expectedDelay := 50 // milliseconds
	if res.BackpressureDelay != expectedDelay {
		t.Fatalf("expected delay = %dms for 80%% usage, got %dms", expectedDelay, res.BackpressureDelay)
	}
}

// Pure cache hits (zero input/output but cache_read or cache_creation > 0)
// must be recorded so quota accounting can bill prompt-cache cost.
func TestShouldRecordUsage_PureCacheHitNotSkipped(t *testing.T) {
	cases := []struct {
		name                                      string
		inputTokens, outputTokens                 int
		cacheReadTokens, cacheCreationTokens      int
		recordZeroToken, hasToolCosts             bool
		want                                      bool
	}{
		{"pure cache_read hit", 0, 0, 5000, 0, false, false, true},
		{"pure cache_creation", 0, 0, 0, 3000, false, false, true},
		{"both cache classes", 0, 0, 100, 200, false, false, true},
		{"normal call", 50, 30, 0, 0, false, false, true},
		{"zero across the board", 0, 0, 0, 0, false, false, false},
		{"zero with tool costs", 0, 0, 0, 0, false, true, true},
		{"zero with audit flag", 0, 0, 0, 0, true, false, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := shouldRecordUsage(
				tc.inputTokens, tc.outputTokens,
				tc.cacheReadTokens, tc.cacheCreationTokens,
				tc.recordZeroToken, tc.hasToolCosts,
			)
			if got != tc.want {
				t.Errorf("shouldRecordUsage(...) = %v, want %v", got, tc.want)
			}
		})
	}
}

// Test that circuit breaker in open state blocks requests via activity
func TestCheckTokenBudgetWithCircuitBreaker_OpenBlocks(t *testing.T) {
	mgr := budget.NewBudgetManager(nil, zap.NewNop())
	acts := NewBudgetActivitiesWithManager(mgr, zap.NewNop())

	userID := "u-breaker"

	// Configure and trip the circuit breaker
	mgr.ConfigureCircuitBreaker(userID, budget.CircuitBreakerConfig{
		FailureThreshold: 1,
		ResetTimeout:     time.Second,
		HalfOpenRequests: 1,
	})
	mgr.RecordFailure(userID)

	in := BudgetCheckInput{
		UserID:          userID,
		SessionID:       "s",
		TaskID:          "task",
		EstimatedTokens: 10,
	}

	res, err := acts.CheckTokenBudgetWithCircuitBreaker(context.Background(), in)
	if err != nil {
		t.Fatalf("activity error: %v", err)
	}
	if res.CanProceed || !res.CircuitBreakerOpen {
		t.Fatalf("expected breaker to block; got: %+v", res)
	}
}
