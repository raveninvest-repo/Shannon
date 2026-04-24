"""Defensive uniform-TTL tests for _force_uniform_cache_ttl.

Guards against the 'messages.N.cache_control.ttl=1h after ttl=5m' 400 error
caused when upstream code (agent.py agent loop, history replay) injects
cache_control with a hardcoded TTL that doesn't match the TTL resolved for
the current request's cache_source.

Anthropic requires TTLs to be monotonic non-increasing across the fixed
processing order: tools -> system -> messages. This guard forces a single
uniform TTL on all cache_control blocks before the request leaves the
provider, making mixed-TTL 400s structurally impossible.
"""

import copy
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-unit-tests")

from llm_provider.anthropic_provider import (
    AnthropicProvider,
    CACHE_TTL_LONG,
    CACHE_TTL_SHORT,
)
from llm_provider.base import CompletionRequest


def _collect_ttls(api_request: dict) -> list:
    """Flatten every cache_control value in an api_request for assertions."""
    found: list = []
    for t in api_request.get("tools", []) or []:
        if isinstance(t, dict) and "cache_control" in t:
            found.append(("tool", t["cache_control"]))
    sys = api_request.get("system")
    if isinstance(sys, list):
        for b in sys:
            if isinstance(b, dict) and "cache_control" in b:
                found.append(("system", b["cache_control"]))
    for m in api_request.get("messages", []) or []:
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and "cache_control" in b:
                    found.append(("message", b["cache_control"]))
    return found


def _mixed_request() -> dict:
    """Reproduce the real bug: agent.py injects 1h on a message, while
    provider has resolved short TTL for system/tools based on cache_source."""
    return {
        "tools": [
            {"name": "web_search", "cache_control": copy.deepcopy(CACHE_TTL_SHORT)},
        ],
        "system": [
            {"type": "text", "text": "sys", "cache_control": copy.deepcopy(CACHE_TTL_SHORT)},
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "turn1"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            # Simulates agent.py:2041 hardcoded CACHE_TTL_LONG injection
            {"role": "user", "content": [
                {"type": "text", "text": "turn2", "cache_control": copy.deepcopy(CACHE_TTL_LONG)},
            ]},
        ],
    }


class TestUniformTTL:
    """Short-source requests get everything forced to 5m."""

    def test_short_source_coerces_hardcoded_1h_to_5m(self):
        req = _mixed_request()
        AnthropicProvider._force_uniform_cache_ttl(req, CACHE_TTL_SHORT)
        ttls = _collect_ttls(req)
        assert ttls, "expected at least one cache_control to remain"
        # Every cache_control must equal CACHE_TTL_SHORT — never mixed.
        for _loc, cc in ttls:
            assert cc == CACHE_TTL_SHORT

    def test_long_source_coerces_hardcoded_5m_to_1h(self):
        req = {
            "tools": [{"name": "t", "cache_control": copy.deepcopy(CACHE_TTL_SHORT)}],
            "system": [{"type": "text", "text": "s", "cache_control": copy.deepcopy(CACHE_TTL_SHORT)}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "u", "cache_control": copy.deepcopy(CACHE_TTL_SHORT)},
                ]},
            ],
        }
        AnthropicProvider._force_uniform_cache_ttl(req, CACHE_TTL_LONG)
        for _loc, cc in _collect_ttls(req):
            assert cc == CACHE_TTL_LONG


class TestForceOff:
    """ttl_block=None (SHANNON_FORCE_TTL=off) strips all cache_control."""

    def test_off_removes_every_cache_control(self):
        req = _mixed_request()
        AnthropicProvider._force_uniform_cache_ttl(req, None)
        assert _collect_ttls(req) == []


class TestNoopWhenAlreadyUniform:
    """Already-uniform requests stay byte-equal after normalization."""

    def test_uniform_short_is_byte_stable(self):
        req = {
            "tools": [{"name": "t", "cache_control": copy.deepcopy(CACHE_TTL_SHORT)}],
            "system": [{"type": "text", "text": "s", "cache_control": copy.deepcopy(CACHE_TTL_SHORT)}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "u", "cache_control": copy.deepcopy(CACHE_TTL_SHORT)},
                ]},
            ],
        }
        before = copy.deepcopy(req)
        AnthropicProvider._force_uniform_cache_ttl(req, CACHE_TTL_SHORT)
        assert req == before


class TestMonotonicOrderInvariant:
    """The real Anthropic constraint: no 1h appearing after a 5m across
    tools->system->messages. Normalization trivially satisfies this by
    producing exactly one unique TTL."""

    def test_real_bug_scenario_single_unique_ttl(self):
        req = _mixed_request()
        AnthropicProvider._force_uniform_cache_ttl(req, CACHE_TTL_SHORT)
        unique = {repr(cc) for _loc, cc in _collect_ttls(req)}
        assert len(unique) == 1, f"expected 1 unique TTL, got {unique}"


class TestBlocksWithoutCacheControlUntouched:
    """Content blocks without cache_control must not gain one."""

    def test_blocks_without_cache_control_stay_without(self):
        req = {
            "tools": [{"name": "t"}],  # no cache_control
            "system": [{"type": "text", "text": "s"}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "u"}]}],
        }
        AnthropicProvider._force_uniform_cache_ttl(req, CACHE_TTL_LONG)
        assert _collect_ttls(req) == []


# --- End-to-end regression: the actual bug path -----------------------------
#
# These tests exercise _build_api_request, not _force_uniform_cache_ttl in
# isolation. They reproduce the real production failure:
# `/agent/query` agent loop, cache_source='agent_execute' (short), second
# iteration where agent.py injects CACHE_TTL_LONG on the last user message.
# Pre-fix these produced a request with mixed 1h/5m → Anthropic 400.

_MINIMAL_CONFIG = {
    "api_key": "test-key",
    "models": {
        "claude-sonnet-4-6": {
            "model_id": "claude-sonnet-4-6",
            "tier": "medium",
            "context_window": 200000,
            "max_tokens": 8192,
        },
    },
}


def _model_config():
    return type("MC", (), {
        "model_id": "claude-sonnet-4-6",
        "supports_functions": False,
        "context_window": 200000,
        "max_tokens": 8192,
    })()


class TestAgentLoopSecondIterationBugE2E:
    """Exercise the full _build_api_request pipeline with the exact layout that
    triggered Anthropic 400 pre-fix."""

    def _make_provider(self):
        return AnthropicProvider(_MINIMAL_CONFIG)

    def _agent_loop_request(self, cache_source: str):
        """Reproduces agent.py:2041/2046: second iteration pastes CACHE_TTL_LONG
        on the last user message while other turns stay plain."""
        return CompletionRequest(
            messages=[
                {"role": "system", "content": "You are a research agent."},
                {"role": "user", "content": "Research X"},
                {"role": "assistant", "content": "I'll search."},
                # agent.py injection: list-form user block with hardcoded 1h
                {"role": "user", "content": [
                    {"type": "text", "text": "Tool result...", "cache_control": CACHE_TTL_LONG},
                ]},
            ],
            temperature=0.3,
            max_tokens=100,
            cache_source=cache_source,
        )

    def test_short_source_agent_execute_normalizes_to_5m(self):
        """Real bug: cache_source='agent_execute' resolves to 5m but agent.py
        injected 1h on messages. Post-fix all cache_control must be uniform."""
        provider = self._make_provider()
        api_req = provider._build_api_request(
            self._agent_loop_request("agent_execute"), _model_config(),
        )
        blocks = _collect_ttls(api_req)
        assert blocks, "expected at least one cache_control block on the request"
        assert all(cc == CACHE_TTL_SHORT for _loc, cc in blocks), (
            f"expected every cache_control == CACHE_TTL_SHORT after normalization; "
            f"got {blocks}. This is the exact condition that produced Anthropic 400 pre-fix."
        )

    def test_long_source_shanclaw_normalizes_to_1h(self):
        """Symmetric case: shanclaw resolves to 1h, and if anything in the
        request was accidentally 5m it should be lifted to 1h."""
        provider = self._make_provider()
        api_req = provider._build_api_request(
            self._agent_loop_request("shanclaw"), _model_config(),
        )
        blocks = _collect_ttls(api_req)
        assert blocks, "expected at least one cache_control block"
        assert all(cc == CACHE_TTL_LONG for _loc, cc in blocks), \
            f"expected uniform CACHE_TTL_LONG; got {blocks}"

    def test_force_off_env_strips_all_cache_control_end_to_end(self, monkeypatch):
        """SHANNON_FORCE_TTL=off must produce a request with zero
        cache_control blocks, regardless of agent.py injections upstream."""
        monkeypatch.setenv("SHANNON_FORCE_TTL", "off")
        provider = self._make_provider()
        api_req = provider._build_api_request(
            self._agent_loop_request("agent_execute"), _model_config(),
        )
        assert _collect_ttls(api_req) == []


class TestMarkLastBlockDedupInteraction:
    """When an upstream caller pre-injects cache_control on a message, the
    provider's _mark_last_block de-dup logic skips its usual [-2] placement.
    After this PR the final TTL is still uniform; this test pins that
    invariant so the implicit coupling doesn't silently regress."""

    def test_caller_preset_cache_control_keeps_single_ttl(self):
        provider = AnthropicProvider(_MINIMAL_CONFIG)
        # Caller pre-pastes cache_control on the SECOND user message — not
        # the penultimate. Pre-fix that would co-exist with provider's own
        # [-2] cache_control in a different TTL; post-fix all are uniform.
        request = CompletionRequest(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Turn 1 question"},
                {"role": "assistant", "content": "Turn 1 answer"},
                {"role": "user", "content": [
                    {"type": "text", "text": "Turn 2 preset by caller",
                     "cache_control": CACHE_TTL_LONG},
                ]},
                {"role": "assistant", "content": "Turn 2 answer"},
                {"role": "user", "content": "Turn 3 latest"},
            ],
            temperature=0.0,
            max_tokens=100,
            cache_source="agent_execute",  # resolves to 5m
        )
        api_req = provider._build_api_request(request, _model_config())
        blocks = _collect_ttls(api_req)
        assert blocks, "expected cache_control blocks from provider writes"
        # With agent_execute (5m) and caller-injected 1h, the guard must
        # collapse everything to the single resolved ttl (5m).
        assert all(cc == CACHE_TTL_SHORT for _loc, cc in blocks), \
            f"dedup + normalization should yield uniform CACHE_TTL_SHORT; got {blocks}"

    def test_modified_counter_fires_only_when_actually_mismatched(self):
        """_force_uniform_cache_ttl must be a no-op (no modifications) when
        caller-side cache_control already matches the resolved ttl_block."""
        import copy
        req = {
            "tools": [{"name": "t", "cache_control": copy.deepcopy(CACHE_TTL_SHORT)}],
            "system": [{"type": "text", "text": "s", "cache_control": copy.deepcopy(CACHE_TTL_SHORT)}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "x", "cache_control": copy.deepcopy(CACHE_TTL_SHORT)},
            ]}],
        }
        before = copy.deepcopy(req)
        AnthropicProvider._force_uniform_cache_ttl(req, CACHE_TTL_SHORT)
        # Byte-equal: guard didn't rewrite identical values (checked via
        # the != comparison in the implementation — test locks it in).
        assert req == before
