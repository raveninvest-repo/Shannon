"""TTL routing tests for _ttl_block.

Precedence: SHANNON_FORCE_TTL env > cache_source map > short default.
cache_ttl field is intentionally NOT consulted for prompt-cache routing
(that field drives manager.py's response-cache layer, a different concern).
"""

import os

# Set dummy key before import
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-unit-tests")

from llm_provider.anthropic_provider import (
    _ttl_block,
    CACHE_TTL_LONG,
    CACHE_TTL_SHORT,
)
from llm_provider.base import CompletionRequest


def _mk(source=None):
    return CompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-4-6",
        cache_source=source,
    )


class TestSourceMapping:
    """Source -> TTL lookup without env overrides."""

    def test_slack_channel_uses_long_ttl(self):
        assert _ttl_block(_mk("slack")) == CACHE_TTL_LONG

    def test_line_channel_uses_long_ttl(self):
        assert _ttl_block(_mk("line")) == CACHE_TTL_LONG

    def test_feishu_channel_uses_long_ttl(self):
        assert _ttl_block(_mk("feishu")) == CACHE_TTL_LONG

    def test_tui_uses_long_ttl(self):
        assert _ttl_block(_mk("tui")) == CACHE_TTL_LONG

    def test_webhook_uses_short_ttl(self):
        assert _ttl_block(_mk("webhook")) == CACHE_TTL_SHORT

    def test_cron_uses_short_ttl(self):
        assert _ttl_block(_mk("cron")) == CACHE_TTL_SHORT

    def test_mcp_uses_short_ttl(self):
        assert _ttl_block(_mk("mcp")) == CACHE_TTL_SHORT

    def test_swarm_subagent_uses_short_ttl(self):
        assert _ttl_block(_mk("swarm_subagent")) == CACHE_TTL_SHORT

    def test_tool_select_uses_short_ttl(self):
        """Internal swarm decision prompts — different each call, no resume."""
        assert _ttl_block(_mk("tool_select")) == CACHE_TTL_SHORT

    def test_lead_decide_uses_short_ttl(self):
        assert _ttl_block(_mk("lead_decide")) == CACHE_TTL_SHORT

    def test_interpretation_uses_short_ttl(self):
        assert _ttl_block(_mk("interpretation")) == CACHE_TTL_SHORT

    def test_unknown_source_defaults_short_fail_cheap(self):
        """Fail-cheap: unrecognized source pays 1.25x premium, not 2x."""
        assert _ttl_block(_mk("some_new_path_we_havent_classified_yet")) == CACHE_TTL_SHORT

    def test_none_source_defaults_short(self):
        assert _ttl_block(_mk(None)) == CACHE_TTL_SHORT

    def test_empty_string_source_defaults_short(self):
        assert _ttl_block(_mk("")) == CACHE_TTL_SHORT

    def test_source_is_case_insensitive(self):
        assert _ttl_block(_mk("SLACK")) == CACHE_TTL_LONG
        assert _ttl_block(_mk("Slack")) == CACHE_TTL_LONG


class TestEnvEscapeHatch:
    """SHANNON_FORCE_TTL overrides source map — operator debug/A-B."""

    def test_force_off_returns_none_suppresses_cache(self, monkeypatch):
        monkeypatch.setenv("SHANNON_FORCE_TTL", "off")
        assert _ttl_block(_mk("slack")) is None
        assert _ttl_block(_mk("webhook")) is None
        assert _ttl_block(_mk(None)) is None

    def test_force_5m_overrides_long_sources(self, monkeypatch):
        monkeypatch.setenv("SHANNON_FORCE_TTL", "5m")
        assert _ttl_block(_mk("slack")) == CACHE_TTL_SHORT
        assert _ttl_block(_mk("tui")) == CACHE_TTL_SHORT

    def test_force_1h_overrides_short_sources(self, monkeypatch):
        """For comparison benches — force everyone to 1h to recreate legacy behavior."""
        monkeypatch.setenv("SHANNON_FORCE_TTL", "1h")
        assert _ttl_block(_mk("webhook")) == CACHE_TTL_LONG
        assert _ttl_block(_mk("cron")) == CACHE_TTL_LONG

    def test_force_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("SHANNON_FORCE_TTL", "OFF")
        assert _ttl_block(_mk("slack")) is None

    def test_force_invalid_value_falls_through_to_source_map(self, monkeypatch):
        """Typos don't silently break routing — invalid env value is ignored."""
        monkeypatch.setenv("SHANNON_FORCE_TTL", "bogus")
        assert _ttl_block(_mk("slack")) == CACHE_TTL_LONG
        assert _ttl_block(_mk("webhook")) == CACHE_TTL_SHORT


class TestCacheTtlFieldIgnored:
    """request.cache_ttl is the response-cache TTL (manager.py), not prompt-cache.
    Prompt-cache routing must ignore it to avoid conflating the two layers."""

    def test_explicit_cache_ttl_does_not_override_source(self):
        req = CompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-6",
            cache_source="webhook",
            cache_ttl=3600,  # would imply 1h under naive interpretation
        )
        # webhook source still routes to short regardless of cache_ttl=3600
        assert _ttl_block(req) == CACHE_TTL_SHORT
