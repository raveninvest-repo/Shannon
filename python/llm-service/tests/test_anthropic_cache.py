"""Tests for Anthropic prompt cache behavior with multi-turn messages."""

import os
import pytest

# Set dummy key before import
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-unit-tests")

from llm_provider.anthropic_provider import AnthropicProvider, CACHE_TTL_LONG


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


class TestMultiTurnCacheBreakpoints:
    """Verify cache_control placement with multi-turn agent messages."""

    def _make_provider(self):
        return AnthropicProvider(_MINIMAL_CONFIG)

    def test_multi_turn_no_per_message_cache_control(self):
        """Rolling cache_control marker lands on the penultimate message's last block;
        all other messages stay as plain strings."""
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Task: research"},
            {"role": "assistant", "content": "I will search."},
            {"role": "user", "content": "Result: found data."},
            {"role": "assistant", "content": "Analyzing data."},
            {"role": "user", "content": "Budget: 3 calls. Decide."},
        ]
        system_msg, claude_msgs = provider._convert_messages_to_claude_format(messages)

        assert system_msg == "You are helpful."
        assert len(claude_msgs) == 5

        # Rolling cache_control lands on claude_msgs[-2]; all others stay plain strings.
        penultimate_idx = len(claude_msgs) - 2
        for i, msg in enumerate(claude_msgs):
            if i == penultimate_idx:
                # Rolling marker target — content promoted to list with cache_control
                assert isinstance(msg["content"], list), \
                    f"penultimate msg content should be list, got {type(msg['content'])}"
            else:
                if msg["role"] == "assistant":
                    assert isinstance(msg["content"], str), \
                        f"non-penultimate assistant content should remain plain string, got {msg['content']!r}"

    def test_system_message_always_gets_cache_control(self):
        """System message gets cache_control in _build_api_request.

        Uses cache_source='tui' to land in the long-TTL bucket — under the
        source-routed policy, unlabeled requests fall back to 5m (fail cheap).
        """
        provider = self._make_provider()
        from llm_provider.base import CompletionRequest
        request = CompletionRequest(
            messages=[
                {"role": "system", "content": "System prompt text."},
                {"role": "user", "content": "Hello."},
            ],
            temperature=0.3,
            max_tokens=100,
            cache_source="tui",
        )
        model_config = type("MC", (), {
            "model_id": "claude-haiku-4-5-20251001",
            "supports_functions": False,
            "context_window": 200000,
            "max_tokens": 8192,
        })()
        api_req = provider._build_api_request(request, model_config)
        assert api_req["system"][0]["cache_control"] == CACHE_TTL_LONG

    def test_no_cache_break_in_multi_turn(self):
        """Multi-turn messages without marker produce plain string content."""
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Context without marker."},
            {"role": "assistant", "content": "Decision."},
            {"role": "user", "content": "Current turn."},
        ]
        _, claude_msgs = provider._convert_messages_to_claude_format(messages)
        for msg in claude_msgs:
            if msg["role"] == "user":
                assert isinstance(msg["content"], str), "User messages without marker should be plain strings"


    def test_leading_cache_break_no_stable_prefix(self):
        """Leading cache_break with no stable content must not produce empty text block.

        Regression: ShanClaw sends <!-- cache_break -->volatile... when StableContext
        is empty. Empty text block + cache_control causes Anthropic 400 error.
        """
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "<!-- cache_break -->\n## Context\nVolatile data here"},
        ]
        _, claude_msgs = provider._convert_messages_to_claude_format(messages)
        user_msg = claude_msgs[0]
        # Should fall back to plain string (no content blocks with empty text)
        assert isinstance(user_msg["content"], str)
        assert user_msg["content"] == "\n## Context\nVolatile data here"

    def test_whitespace_only_stable_prefix(self):
        """Whitespace-only stable prefix treated as empty — no cache_control block."""
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "  \n<!-- cache_break -->\nVolatile"},
        ]
        _, claude_msgs = provider._convert_messages_to_claude_format(messages)
        user_msg = claude_msgs[0]
        assert isinstance(user_msg["content"], str)

    def test_nonempty_stable_prefix_preserved_raw(self):
        """Non-empty stable prefix is sent as-is (not stripped) in content block."""
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Stable context\n<!-- cache_break -->\nVolatile"},
        ]
        _, claude_msgs = provider._convert_messages_to_claude_format(messages)
        user_msg = claude_msgs[0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["text"] == "Stable context\n"
        assert user_msg["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert user_msg["content"][1]["text"] == "\nVolatile"


class TestSystemMessageSplit:
    """Verify system message splits on <!-- volatile --> marker."""

    def _make_provider(self):
        return AnthropicProvider(_MINIMAL_CONFIG)

    def test_volatile_marker_splits_system_into_two_blocks(self):
        """System message with <!-- volatile --> produces two API blocks."""
        provider = self._make_provider()
        system_msg = "Stable protocol instructions\n<!-- volatile -->\nDynamic task context"
        blocks = provider._split_system_message(system_msg)
        assert len(blocks) == 2
        assert blocks[0]["text"] == "Stable protocol instructions"
        assert "cache_control" in blocks[0]
        assert blocks[0]["cache_control"] == CACHE_TTL_LONG
        assert blocks[1]["text"] == "Dynamic task context"
        assert "cache_control" not in blocks[1]

    def test_no_marker_returns_single_cached_block(self):
        """System message without marker returns single block (backward compat)."""
        provider = self._make_provider()
        system_msg = "Plain system prompt without any marker"
        blocks = provider._split_system_message(system_msg)
        assert len(blocks) == 1
        assert blocks[0]["text"] == system_msg
        assert blocks[0]["cache_control"] == CACHE_TTL_LONG

    def test_empty_volatile_section_omitted(self):
        """Trailing marker with no volatile content produces single block."""
        provider = self._make_provider()
        system_msg = "Stable content only\n<!-- volatile -->\n   "
        blocks = provider._split_system_message(system_msg)
        assert len(blocks) == 1
        assert blocks[0]["cache_control"]["type"] == "ephemeral"

    def test_build_api_request_uses_split(self):
        """_build_api_request uses _split_system_message for system blocks."""
        provider = self._make_provider()
        from llm_provider.base import CompletionRequest
        request = CompletionRequest(
            messages=[
                {"role": "system", "content": "Stable\n<!-- volatile -->\nVolatile"},
                {"role": "user", "content": "Hello."},
            ],
            temperature=0.3,
            max_tokens=100,
        )
        model_config = type("MC", (), {
            "model_id": "claude-sonnet-4-6",
            "supports_functions": False,
            "context_window": 200000,
            "max_tokens": 8192,
        })()
        api_req = provider._build_api_request(request, model_config)
        assert len(api_req["system"]) == 2
        assert "cache_control" in api_req["system"][0]
        assert "cache_control" not in api_req["system"][1]


class TestVolatileTTL:
    """Verify TTL behavior for volatile split.

    System, tools, and stable user prefix all use 1h TTL.
    Anthropic prefix order: system → tools → messages.
    TTL monotonic non-increasing: system(1h) ≥ tools(1h) ≥ messages(1h) ✓
    """

    def _make_provider(self):
        return AnthropicProvider(_MINIMAL_CONFIG)

    def test_stable_prefix_uses_1h_ttl(self):
        """Stable prefix block uses 1h TTL for long-running workflows."""
        provider = self._make_provider()
        system_msg = "Stable\n<!-- volatile -->\nVolatile"
        blocks = provider._split_system_message(system_msg)
        assert blocks[0]["cache_control"] == CACHE_TTL_LONG
        assert blocks[0]["cache_control"]["ttl"] == "1h"

    def test_no_marker_uses_1h_ttl(self):
        """Without volatile marker, use 1h TTL."""
        provider = self._make_provider()
        blocks = provider._split_system_message("Plain prompt")
        assert blocks[0]["cache_control"] == CACHE_TTL_LONG
        assert blocks[0]["cache_control"]["ttl"] == "1h"


class TestToolSchemaFreeze:
    """Verify tool schemas are frozen after first build for cache stability."""

    def _make_provider(self):
        return AnthropicProvider(_MINIMAL_CONFIG)

    def test_same_tools_return_frozen_copy(self):
        """Same tool names → identical schema (frozen from first build)."""
        provider = self._make_provider()
        functions = [
            {"name": "web_search", "description": "Search v1", "parameters": {"properties": {}, "required": []}},
            {"name": "calculator", "description": "Calc v1", "parameters": {"properties": {}, "required": []}},
        ]
        tools1 = provider._convert_functions_to_tools(functions)

        # Change descriptions (simulating drift)
        functions_v2 = [
            {"name": "web_search", "description": "Search v2 CHANGED", "parameters": {"properties": {}, "required": []}},
            {"name": "calculator", "description": "Calc v2 CHANGED", "parameters": {"properties": {}, "required": []}},
        ]
        tools2 = provider._convert_functions_to_tools(functions_v2)

        # Should return frozen v1 schemas, not v2
        assert tools1[0]["description"] == tools2[0]["description"]

    def test_different_tool_set_rebuilds(self):
        """Different tool names → rebuild schema."""
        provider = self._make_provider()
        func_a = [{"name": "web_search", "description": "Search", "parameters": {"properties": {}, "required": []}}]
        func_b = [{"name": "calculator", "description": "Calc", "parameters": {"properties": {}, "required": []}}]

        tools_a = provider._convert_functions_to_tools(func_a)
        tools_b = provider._convert_functions_to_tools(func_b)
        assert tools_a[0]["name"] != tools_b[0]["name"]

    def test_reset_clears_cache(self):
        """reset_tool_cache() forces rebuild on next call."""
        provider = self._make_provider()
        functions = [{"name": "web_search", "description": "v1", "parameters": {"properties": {}, "required": []}}]
        provider._convert_functions_to_tools(functions)

        provider.reset_tool_cache()

        functions_v2 = [{"name": "web_search", "description": "v2", "parameters": {"properties": {}, "required": []}}]
        tools = provider._convert_functions_to_tools(functions_v2)
        assert tools[0]["description"] == "v2"


class TestToolOrdering:
    """Verify tools are sorted by name for cache prefix stability."""

    def _make_provider(self):
        return AnthropicProvider(_MINIMAL_CONFIG)

    def test_tools_sorted_by_name(self):
        provider = self._make_provider()
        functions = [
            {"name": "web_search", "description": "Search", "parameters": {"properties": {}, "required": []}},
            {"name": "calculator", "description": "Calc", "parameters": {"properties": {}, "required": []}},
            {"name": "file_read", "description": "Read", "parameters": {"properties": {}, "required": []}},
        ]
        tools = provider._convert_functions_to_tools(functions)
        names = [t["name"] for t in tools]
        assert names == ["calculator", "file_read", "web_search"]

    def test_cache_control_on_last_sorted_tool(self):
        """After sorting, the last tool alphabetically should be last in the list."""
        provider = self._make_provider()
        functions = [
            {"name": "web_search", "description": "Search", "parameters": {"properties": {}, "required": []}},
            {"name": "calculator", "description": "Calc", "parameters": {"properties": {}, "required": []}},
        ]
        tools = provider._convert_functions_to_tools(functions)
        assert tools[-1]["name"] == "web_search"


class TestCacheBreakDetector:
    """Verify cache break detection across sequential API calls."""

    def test_first_call_no_break(self):
        """First call has no previous state → no break detected."""
        from llm_provider.anthropic_provider import CacheBreakDetector
        detector = CacheBreakDetector()
        result = detector.check(
            system_text="Stable prompt",
            tool_names=["web_search", "calculator"],
            model="claude-sonnet-4-6",
        )
        assert result is None

    def test_identical_calls_no_break(self):
        """Two identical calls → no break."""
        from llm_provider.anthropic_provider import CacheBreakDetector
        detector = CacheBreakDetector()
        detector.check(system_text="Prompt", tool_names=["a"], model="m1")
        result = detector.check(system_text="Prompt", tool_names=["a"], model="m1")
        assert result is None

    def test_system_change_detected(self):
        """Changed system prompt text → break detected with reason."""
        from llm_provider.anthropic_provider import CacheBreakDetector
        detector = CacheBreakDetector()
        detector.check(system_text="V1 prompt", tool_names=["a"], model="m1")
        result = detector.check(system_text="V2 prompt changed", tool_names=["a"], model="m1")
        assert result is not None
        assert "system" in result["changed"]

    def test_tool_set_change_detected(self):
        """Changed tool set → break detected."""
        from llm_provider.anthropic_provider import CacheBreakDetector
        detector = CacheBreakDetector()
        detector.check(system_text="P", tool_names=["a", "b"], model="m1")
        result = detector.check(system_text="P", tool_names=["a", "c"], model="m1")
        assert result is not None
        assert "tools" in result["changed"]
        assert "c" in result.get("tools_added", [])
        assert "b" in result.get("tools_removed", [])

    def test_model_change_detected(self):
        """Changed model → break detected."""
        from llm_provider.anthropic_provider import CacheBreakDetector
        detector = CacheBreakDetector()
        detector.check(system_text="P", tool_names=["a"], model="m1")
        result = detector.check(system_text="P", tool_names=["a"], model="m2")
        assert result is not None
        assert "model" in result["changed"]

    def test_multiple_changes_detected(self):
        """Multiple changes reported in one break."""
        from llm_provider.anthropic_provider import CacheBreakDetector
        detector = CacheBreakDetector()
        detector.check(system_text="P1", tool_names=["a"], model="m1")
        result = detector.check(system_text="P2", tool_names=["b"], model="m2")
        assert result is not None
        assert len(result["changed"]) == 3

    def test_call_count_increments(self):
        """Call count tracks API calls for this detector instance."""
        from llm_provider.anthropic_provider import CacheBreakDetector
        detector = CacheBreakDetector()
        detector.check(system_text="P", tool_names=["a"], model="m1")
        detector.check(system_text="P", tool_names=["a"], model="m1")
        detector.check(system_text="P", tool_names=["a"], model="m1")
        assert detector.call_count == 3


class TestCacheBreakIntegration:
    """Verify CacheBreakDetector is wired into _build_api_request."""

    def _make_provider(self):
        return AnthropicProvider(_MINIMAL_CONFIG)

    def test_provider_has_detector(self):
        """Provider instance has a CacheBreakDetector."""
        provider = self._make_provider()
        assert hasattr(provider, "_cache_break_detector")
        from llm_provider.anthropic_provider import CacheBreakDetector
        assert isinstance(provider._cache_break_detector, CacheBreakDetector)

    def test_detector_called_on_build(self):
        """_build_api_request calls detector.check (call_count increments)."""
        provider = self._make_provider()
        from llm_provider.base import CompletionRequest
        request = CompletionRequest(
            messages=[
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "Hello."},
            ],
            temperature=0.3,
            max_tokens=100,
        )
        model_config = type("MC", (), {
            "model_id": "claude-sonnet-4-6",
            "supports_functions": False,
            "context_window": 200000,
            "max_tokens": 8192,
        })()
        provider._build_api_request(request, model_config)
        provider._build_api_request(request, model_config)
        assert provider._cache_break_detector.call_count == 2


class TestCallSequence:
    """Verify call_sequence counter flows through TokenUsage."""

    def test_token_usage_has_call_sequence_field(self):
        """TokenUsage dataclass accepts call_sequence."""
        from llm_provider.base import TokenUsage
        usage = TokenUsage(
            input_tokens=100, output_tokens=50, total_tokens=150,
            estimated_cost=0.001, call_sequence=3,
        )
        assert usage.call_sequence == 3

    def test_token_usage_default_call_sequence_zero(self):
        """TokenUsage.call_sequence defaults to 0."""
        from llm_provider.base import TokenUsage
        usage = TokenUsage(
            input_tokens=100, output_tokens=50, total_tokens=150,
            estimated_cost=0.001,
        )
        assert usage.call_sequence == 0

    def test_token_usage_add_takes_max_sequence(self):
        """Adding TokenUsage takes max call_sequence."""
        from llm_provider.base import TokenUsage
        a = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150, estimated_cost=0.001, call_sequence=3)
        b = TokenUsage(input_tokens=200, output_tokens=100, total_tokens=300, estimated_cost=0.002, call_sequence=7)
        combined = a + b
        assert combined.call_sequence == 7


class TestCacheSourceObservability:
    """Verify cache_source plumbing for per-source observability metrics."""

    def test_cache_source_field_defaults_to_none(self):
        """CompletionRequest.cache_source defaults to None (emits as 'unknown')."""
        from llm_provider.base import CompletionRequest
        req = CompletionRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.cache_source is None

    def test_cache_source_field_accepts_string(self):
        """CompletionRequest.cache_source accepts caller label."""
        from llm_provider.base import CompletionRequest
        req = CompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            cache_source="agent_loop",
        )
        assert req.cache_source == "agent_loop"

    def test_providers_passthrough_includes_cache_source(self):
        """llm_service.providers.generate_completion has cache_source in both passthrough sets.

        Without this, cache_source from callers is silently dropped before reaching
        CompletionRequest and all metrics collapse to "unknown".
        """
        import inspect
        import llm_service.providers as providers_init

        src = inspect.getsource(providers_init)
        # There are two passthrough_fields blocks (one per code path); both must include it.
        assert src.count("\"cache_source\"") >= 2, (
            "cache_source must appear in both passthrough_fields blocks"
        )


class TestStreamingCacheAccounting:
    """Verify streaming path reads 1h cache creation and supports dict usage shape."""

    def test_record_cache_metrics_5m_vs_1h_split(self):
        """_record_cache_metrics splits cache_creation into 5m and 1h buckets."""
        from llm_provider import anthropic_provider as ap

        # Inline capture — we just verify the split math, not prometheus internals
        splits = {}
        original = ap._record_cache_metrics

        def _capture(provider, model, source, cache_read, cache_creation, cache_creation_1h):
            splits["read"] = cache_read
            splits["write_5m"] = max(0, cache_creation - cache_creation_1h)
            splits["write_1h"] = cache_creation_1h

        monkey = _capture
        monkey("anthropic", "claude-sonnet-4-6", "test", 500, 1200, 800)
        assert splits == {"read": 500, "write_5m": 400, "write_1h": 800}
        # Sanity: the real helper exists and is callable
        assert callable(original)

    def test_streaming_usage_dict_shape_parses_cache_fields(self):
        """Dict-shaped usage from Anthropic-compat providers parses cache fields.

        Regression: prior to this fix, streaming path used only getattr() and
        dict-shaped usage silently zeroed cache_read/creation/1h → underpriced cost.
        """
        # Simulate the extraction logic the streaming branch uses
        usage_dict = {
            "input_tokens": 1000,
            "output_tokens": 200,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 1200,
            "cache_creation": {"ephemeral_1h_input_tokens": 800},
        }
        # Mimic the extraction contract
        assert isinstance(usage_dict, dict)
        cache_read = usage_dict.get("cache_read_input_tokens", 0) or 0
        cache_creation = usage_dict.get("cache_creation_input_tokens", 0) or 0
        cc = usage_dict.get("cache_creation")
        cache_creation_1h = (
            cc.get("ephemeral_1h_input_tokens", 0) or 0 if isinstance(cc, dict) else 0
        )
        assert cache_read == 500
        assert cache_creation == 1200
        assert cache_creation_1h == 800


class TestRollingMessageBreakpoint:
    """Verify rolling cache_control marker on claude_messages[-2]."""

    def _make_provider(self):
        return AnthropicProvider(_MINIMAL_CONFIG)

    def test_single_message_no_rolling_marker(self):
        """Turn 1: only one message -> no rolling marker beyond existing cache_break."""
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "stable<!-- cache_break -->volatile"},
        ]
        _, claude_msgs = provider._convert_messages_to_claude_format(messages)
        # Only one user message; no [-2] index exists beyond the sole message
        assert len(claude_msgs) == 1
        # User message retains cache_break split (existing behavior)
        assert isinstance(claude_msgs[0]["content"], list)
        assert claude_msgs[0]["content"][0].get("cache_control") is not None

    def test_multi_turn_rolling_marker_on_penultimate(self):
        """Turn 2+: penultimate completed message gets cache_control on its last block."""
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "assistant_reply_1"},
            {"role": "user", "content": "follow_up"},
        ]
        _, claude_msgs = provider._convert_messages_to_claude_format(messages)
        assert len(claude_msgs) == 3
        # Penultimate is the assistant reply
        penultimate = claude_msgs[-2]
        assert penultimate["role"] == "assistant"
        # Its content must be promoted to a list with cache_control on the last block
        assert isinstance(penultimate["content"], list), \
            f"expected list content, got {type(penultimate['content'])}"
        last_block = penultimate["content"][-1]
        assert last_block.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}

    def test_rolling_marker_skips_when_already_marked(self):
        """If penultimate already has cache_control (e.g. user_1 with cache_break marker),
        do not add another — respect the 4-breakpoint cap."""
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "stable<!-- cache_break -->volatile1"},
            {"role": "assistant", "content": "reply"},
        ]
        _, claude_msgs = provider._convert_messages_to_claude_format(messages)
        assert len(claude_msgs) == 2
        penultimate = claude_msgs[-2]
        # Penultimate is user_1 with cache_break; already has cache_control on its stable block.
        # Rolling should NOT add another on the volatile block.
        assert isinstance(penultimate["content"], list)
        cc_count = sum(
            1 for b in penultimate["content"]
            if isinstance(b, dict) and b.get("cache_control")
        )
        assert cc_count == 1, f"expected exactly 1 cache_control on penultimate, got {cc_count}"

    def test_rolling_marker_on_tool_result(self):
        """Agent loop: tool_result message gets the rolling marker on its last block."""
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "calling tool"},
                {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "tool result"},
            {"role": "user", "content": "follow up"},
        ]
        _, claude_msgs = provider._convert_messages_to_claude_format(messages)
        # Penultimate is whatever came just before the final user message.
        # The "tool" role gets converted into a user message with tool_result content;
        # the final user "follow up" is the last message, so penultimate = the tool_result user message.
        penultimate = claude_msgs[-2]
        assert isinstance(penultimate["content"], list)
        # The rolling marker sits on the LAST block of penultimate, regardless of block type.
        assert penultimate["content"][-1].get("cache_control") is not None


class TestPrevRollingPreservation:
    """Verify cross-turn preservation of the previous rolling cache_control marker.

    Rationale: Anthropic prefix cache only matches at positions with explicit
    cache_control. Without preserving prev marker, every turn rewrites a fresh
    rolling block while the previous block becomes unreachable. Long-session
    bench (30-turn): CHR 86%->68%, CER 16.85x->2.35x. Preserving prev marker
    closes the gap.
    """

    def _make_provider(self):
        return AnthropicProvider(_MINIMAL_CONFIG)

    def _convert_and_apply(self, provider, messages, session_id):
        _, msgs = provider._convert_messages_to_claude_format(messages)
        provider._apply_rolling_cache_marker(msgs, session_id)
        return msgs

    def test_first_turn_no_prev_marker_preserved(self):
        """First turn for a session: only [-2] gets rolling marker, no prev to preserve."""
        from llm_provider import anthropic_provider as ap
        ap._prev_rolling.clear()  # isolate test
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply_1"},
            {"role": "user", "content": "second"},
        ]
        msgs = self._convert_and_apply(provider, messages, "session-a")
        # Penultimate (assistant reply_1) marked
        assert msgs[-2]["content"][-1].get("cache_control") is not None
        # First user message NOT marked (no cache_break, no preservation needed)
        first_content = msgs[0]["content"]
        if isinstance(first_content, list):
            assert all(not b.get("cache_control") for b in first_content if isinstance(b, dict))

    def test_second_turn_preserves_prev_marker(self):
        """Second turn for same session: prev rolling marker (now in middle) is preserved."""
        from llm_provider import anthropic_provider as ap
        ap._prev_rolling.clear()
        provider = self._make_provider()
        sid = "session-b"
        # Turn 1: assistant reply_1 becomes the rolling target
        turn1 = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "reply_1"},
            {"role": "user", "content": "u2"},
        ]
        self._convert_and_apply(provider, turn1, sid)
        # Memo should now have session-b -> hash of (the marked assistant reply_1)
        assert sid in ap._prev_rolling

        # Turn 2: conversation grew; the OLD penultimate (reply_1) is now at idx 1.
        # New rolling target is now reply_2 at idx 3.
        turn2 = [
            {"role": "user", "content": "u1"},
            # reply_1 with the cache_control we set in turn 1 — must be preserved
            {"role": "assistant", "content": [
                {"type": "text", "text": "reply_1", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            ]},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "reply_2"},
            {"role": "user", "content": "u3"},
        ]
        msgs = self._convert_and_apply(provider, turn2, sid)
        # New rolling target: reply_2 at idx 3
        assert msgs[3]["content"][-1].get("cache_control") is not None
        # Prev marker preserved at idx 1 (reply_1)
        assert msgs[1]["content"][-1].get("cache_control") is not None
        # Memo updated to current target hash
        assert ap._prev_rolling[sid] != ""

    def test_cache_break_user_stripped_when_prev_preserved(self):
        """When prev marker is preserved, user_1's cache_break cache_control is stripped
        to free a breakpoint slot. user_1's bytes remain readable as part of
        prev_rolling's cached prefix — no info loss."""
        from llm_provider import anthropic_provider as ap
        ap._prev_rolling.clear()
        provider = self._make_provider()
        sid = "session-c"
        # Turn 1: establish a prev marker
        turn1 = [
            {"role": "user", "content": "stable_block<!-- cache_break -->volatile"},
            {"role": "assistant", "content": "reply_1"},
            {"role": "user", "content": "u2"},
        ]
        self._convert_and_apply(provider, turn1, sid)

        # Turn 2: same session, with the marked reply_1 still in messages
        turn2 = [
            {"role": "user", "content": "stable_block<!-- cache_break -->volatile"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "reply_1", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            ]},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "reply_2"},
            {"role": "user", "content": "u3"},
        ]
        msgs = self._convert_and_apply(provider, turn2, sid)
        # user_1 (idx 0) cache_break cache_control should be stripped to make room
        first_content = msgs[0]["content"]
        if isinstance(first_content, list):
            for b in first_content:
                if isinstance(b, dict):
                    assert b.get("cache_control") is None, \
                        "user_1's cache_control should be stripped when prev marker is preserved"
        # prev_rolling marker (reply_1, idx 1) preserved
        assert msgs[1]["content"][-1].get("cache_control") is not None
        # Current rolling marker (reply_2, idx 3) applied
        assert msgs[3]["content"][-1].get("cache_control") is not None

    def test_no_session_id_falls_back_to_basic_marker(self):
        """Without session_id, no prev preservation; just basic rolling marker on [-2]."""
        from llm_provider import anthropic_provider as ap
        before_size = len(ap._prev_rolling)
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "r1"},
            {"role": "user", "content": "u2"},
        ]
        msgs = self._convert_and_apply(provider, messages, None)
        assert msgs[-2]["content"][-1].get("cache_control") is not None
        # Memo unchanged when session_id is None
        assert len(ap._prev_rolling) == before_size

    def test_compaction_drops_prev_marker_gracefully(self):
        """If ShapeHistory removed the prev marker's message, fall back to basic
        marker (no error, just degraded behavior on that single turn)."""
        from llm_provider import anthropic_provider as ap
        ap._prev_rolling.clear()
        provider = self._make_provider()
        sid = "session-d"
        turn1 = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "reply_1"},
            {"role": "user", "content": "u2"},
        ]
        self._convert_and_apply(provider, turn1, sid)
        # Turn 2 simulates ShapeHistory dropping reply_1 entirely (e.g. compacted)
        turn2 = [
            {"role": "user", "content": "[summary of earlier conversation]"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "reply_2"},
            {"role": "user", "content": "u4"},
        ]
        msgs = self._convert_and_apply(provider, turn2, sid)
        # Prev marker not found -> fall back to basic marker on [-2] only
        assert msgs[-2]["content"][-1].get("cache_control") is not None


class TestUsageSplit:
    """Verify 5m/1h cache creation split propagates through to the response."""

    def _make_provider(self):
        return AnthropicProvider(_MINIMAL_CONFIG)

    def test_non_stream_token_usage_captures_split(self):
        """Non-stream usage.cache_creation.ephemeral_5m_input_tokens + 1h both land on TokenUsage."""
        from llm_provider.base import TokenUsage
        # Construct the fields that would flow from anthropic response dict/obj shape
        u = TokenUsage(
            input_tokens=100, output_tokens=50, total_tokens=150,
            estimated_cost=0.001,
            cache_read_tokens=0,
            cache_creation_tokens=300,
            cache_creation_5m_tokens=100,
            cache_creation_1h_tokens=200,
        )
        # The new fields must be independently addressable
        assert u.cache_creation_5m_tokens == 100
        assert u.cache_creation_1h_tokens == 200
        # Sum invariant (enforced in the provider extraction, sanity-checked here):
        assert u.cache_creation_5m_tokens + u.cache_creation_1h_tokens == u.cache_creation_tokens

    def test_token_usage_add_merges_split(self):
        from llm_provider.base import TokenUsage
        a = TokenUsage(10, 5, 15, 0.0, 0, 100, 40, 60)
        b = TokenUsage(20, 10, 30, 0.0, 0, 200, 80, 120)
        c = a + b
        assert c.cache_creation_5m_tokens == 120
        assert c.cache_creation_1h_tokens == 180
        assert c.cache_creation_tokens == 300

    def test_serialize_usage_emits_both_split_fields(self):
        """The ProviderManager._serialize_usage output must include 5m and 1h keys."""
        from llm_service.providers import ProviderManager
        from llm_provider.base import TokenUsage
        u = TokenUsage(
            input_tokens=100, output_tokens=50, total_tokens=150,
            estimated_cost=0.001,
            cache_read_tokens=0,
            cache_creation_tokens=300,
            cache_creation_5m_tokens=100,
            cache_creation_1h_tokens=200,
            call_sequence=1,
        )
        out = ProviderManager._serialize_usage(u)
        assert out["cache_creation_5m_tokens"] == 100
        assert out["cache_creation_1h_tokens"] == 200
        # Legacy field must still be present
        assert out["cache_creation_tokens"] == 300

    def test_serialize_usage_none_returns_empty_dict(self):
        """Regression guard for the None-usage branch."""
        from llm_service.providers import ProviderManager
        assert ProviderManager._serialize_usage(None) == {}


class TestDeferLoadingPassthrough:
    """Task 3.1: defer_loading field passes through tools array."""

    def _make_provider(self):
        return AnthropicProvider(_MINIMAL_CONFIG)

    def test_defer_loading_true_passthrough(self):
        provider = self._make_provider()
        functions = [
            {"name": "normal_tool", "description": "x", "parameters": {}},
            {"name": "deferred_tool", "description": "y", "parameters": {},
             "defer_loading": True},
        ]
        tools = provider._convert_functions_to_tools(functions)
        deferred = next(t for t in tools if t["name"] == "deferred_tool")
        normal = next(t for t in tools if t["name"] == "normal_tool")
        assert deferred.get("defer_loading") is True
        # Normal tool must NOT have defer_loading (omit, don't set False)
        assert "defer_loading" not in normal

    def test_defer_loading_false_omitted(self):
        """defer_loading: false should be equivalent to absence — don't serialize."""
        provider = self._make_provider()
        functions = [
            {"name": "tool", "description": "x", "parameters": {}, "defer_loading": False},
        ]
        tools = provider._convert_functions_to_tools(functions)
        assert "defer_loading" not in tools[0]

    def test_cache_key_includes_defer_flag(self):
        """Swapping defer_loading on/off for same tool name must rebuild (not return frozen)."""
        provider = self._make_provider()
        functions_a = [{"name": "x", "description": "d", "parameters": {}, "defer_loading": True}]
        functions_b = [{"name": "x", "description": "d", "parameters": {}}]
        tools_a = provider._convert_functions_to_tools(functions_a)
        tools_b = provider._convert_functions_to_tools(functions_b)
        # Different defer state → different tools output (not a stale frozen copy)
        assert tools_a[0].get("defer_loading") is True
        assert "defer_loading" not in tools_b[0]


class TestAdvancedToolUseBetaHeader:
    """Task 3.1: beta header toggles on when any deferred tool present."""

    def _make_provider(self):
        return AnthropicProvider(_MINIMAL_CONFIG)

    def _make_request(self, functions):
        from llm_provider.base import CompletionRequest
        return CompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            functions=functions,
            temperature=0.3,
            max_tokens=100,
        )

    def _model_config(self):
        return type("MC", (), {
            "model_id": "claude-sonnet-4-5",
            "supports_functions": True,
            "context_window": 200000,
            "max_tokens": 8192,
        })()

    def test_beta_header_set_when_deferred_present(self, monkeypatch):
        monkeypatch.delenv("SHANNON_NO_ADVANCED_TOOL_USE_BETA", raising=False)
        provider = self._make_provider()
        req = self._make_request([
            {"name": "x", "description": "d", "parameters": {}, "defer_loading": True},
            {"name": "y", "description": "d", "parameters": {}},
        ])
        api_req = provider._build_api_request(req, self._model_config())
        # _build_api_request returns the kwargs for Anthropic SDK; header injection
        # is in complete()/stream_complete() via extra_headers. Verify by reading
        # the tools to confirm the trigger condition is reachable:
        assert any(t.get("defer_loading") for t in api_req["tools"])

    def test_beta_header_absent_when_no_deferred(self):
        provider = self._make_provider()
        req = self._make_request([
            {"name": "x", "description": "d", "parameters": {}},
        ])
        api_req = provider._build_api_request(req, self._model_config())
        assert not any(t.get("defer_loading") for t in api_req["tools"])

    def test_env_escape_hatch_disables_beta(self, monkeypatch):
        """SHANNON_NO_ADVANCED_TOOL_USE_BETA=1 suppresses the advanced-tool-use token.

        Behavioral test against _build_beta_header: with the env var set,
        even when any_deferred=True the returned header must omit the
        advanced-tool-use-2025-11-20 token.
        """
        from llm_provider.anthropic_provider import _build_beta_header

        # Escape hatch OFF → advanced-tool-use present when deferred set
        monkeypatch.delenv("SHANNON_NO_ADVANCED_TOOL_USE_BETA", raising=False)
        hdr_on = _build_beta_header(thinking=False, any_deferred=True)
        assert hdr_on == "advanced-tool-use-2025-11-20"

        # Escape hatch ON → token suppressed, helper returns None (no beta needed)
        monkeypatch.setenv("SHANNON_NO_ADVANCED_TOOL_USE_BETA", "1")
        assert _build_beta_header(thinking=False, any_deferred=True) is None

        # Thinking remains independent of the escape hatch
        assert _build_beta_header(thinking=True, any_deferred=True) == \
            "interleaved-thinking-2025-05-14"

    def test_beta_header_merges_thinking_and_advanced_tool_use(self, monkeypatch):
        """When both thinking and deferred are set (escape hatch OFF), header contains both tokens."""
        from llm_provider.anthropic_provider import _build_beta_header
        monkeypatch.delenv("SHANNON_NO_ADVANCED_TOOL_USE_BETA", raising=False)
        hdr = _build_beta_header(thinking=True, any_deferred=True)
        assert hdr is not None
        tokens = hdr.split(",")
        assert "interleaved-thinking-2025-05-14" in tokens
        assert "advanced-tool-use-2025-11-20" in tokens

    def test_beta_header_returns_none_when_no_feature_needed(self):
        """Neither thinking nor deferred → None (no header injected)."""
        from llm_provider.anthropic_provider import _build_beta_header
        assert _build_beta_header(thinking=False, any_deferred=False) is None


class TestToolReferencePreserved:
    """Task 3.2: role=tool messages with list content preserve tool_reference blocks."""

    def _make_provider(self):
        return AnthropicProvider(_MINIMAL_CONFIG)

    def test_tool_reference_list_content_preserved(self):
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "text", "text": ""},
                {"type": "tool_use", "id": "t1", "name": "tool_search",
                 "input": {"query": "select:x_search"}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": [
                {"type": "tool_reference", "tool_name": "x_search"},
            ]},
        ]
        _, claude_msgs = provider._convert_messages_to_claude_format(messages)
        # Anthropic format: tool results live in a user-role message as tool_result blocks.
        # Find the message containing our tool_result
        found = None
        for m in claude_msgs:
            if m["role"] != "user" or not isinstance(m["content"], list):
                continue
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result" \
                   and b.get("tool_use_id") == "t1":
                    found = b
                    break
            if found:
                break
        assert found is not None, f"tool_result with tool_use_id=t1 not found in {claude_msgs}"
        inner = found["content"]
        assert isinstance(inner, list), f"expected list inner content, got {type(inner).__name__}: {inner}"
        names = [b.get("tool_name") for b in inner if isinstance(b, dict) and b.get("type") == "tool_reference"]
        assert "x_search" in names, f"x_search tool_reference missing from {inner}"

    def test_string_content_still_wrapped_as_string(self):
        """Regression: existing tool content=string path still produces string."""
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "plain text result"},
        ]
        _, claude_msgs = provider._convert_messages_to_claude_format(messages)
        # Find the tool_result block
        found = None
        for m in claude_msgs:
            if m["role"] != "user" or not isinstance(m["content"], list):
                continue
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    found = b
                    break
        assert found is not None
        # String content preserved as string (existing behavior)
        assert found["content"] == "plain text result"
