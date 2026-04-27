package server

import "testing"

func TestEnsureCacheAwareTotal(t *testing.T) {
	tests := []struct {
		name string
		in   RecordUsageDetails
		want int64
	}{
		{
			name: "preserves caller-supplied rollup",
			in: RecordUsageDetails{
				InputTokens:           100,
				OutputTokens:          50,
				CacheReadTokens:       200,
				CacheCreationTokens:   25,
				CacheAwareTotalTokens: 999,
			},
			want: 999,
		},
		{
			name: "computes from parts when zero",
			in: RecordUsageDetails{
				InputTokens:         100,
				OutputTokens:        50,
				CacheReadTokens:     200,
				CacheCreationTokens: 25,
			},
			want: 375,
		},
		{
			name: "cache-only call still recorded",
			in: RecordUsageDetails{
				CacheReadTokens: 1000,
			},
			want: 1000,
		},
		{
			name: "all-zero stays zero so quota guard skips",
			in:   RecordUsageDetails{},
			want: 0,
		},
		{
			name: "negative parts ignored when sum non-positive",
			in: RecordUsageDetails{
				InputTokens:  -10,
				OutputTokens: 5,
			},
			want: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			d := tc.in
			d.EnsureCacheAwareTotal()
			if d.CacheAwareTotalTokens != tc.want {
				t.Fatalf("CacheAwareTotalTokens = %d, want %d", d.CacheAwareTotalTokens, tc.want)
			}
		})
	}
}

func TestEnsureCacheAwareTotal_NilReceiver(t *testing.T) {
	var d *RecordUsageDetails
	d.EnsureCacheAwareTotal()
}
