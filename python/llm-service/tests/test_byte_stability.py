"""Byte-stability regression tests — pin down cross-turn hash invariants so
prev_marker preservation stays reliable.

These tests protect against the pathology observed in session
2026-04-15-69f601dc1c98 (same system_len across 61 requests but 17 distinct
system_h hashes → −13pp CHR vs session-peer median). The stable-hash function
must be invariant under content shape that clients may legitimately vary.
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-unit-tests")

from llm_provider.anthropic_provider import (
    _msg_stable_hash,
    _strip_cache_control_for_hash,
)


class TestStableHashInvariants:
    """Logical equivalents MUST hash identically. Each test case represents a
    serialization shape that can legitimately vary across requests without a
    semantic change."""

    def test_string_content_shape_is_distinguished(self):
        """DOCUMENTS intentional behavior: a string-form message and a
        text-block-form message hash differently. This is correct because the
        two produce different bytes on the Anthropic wire, so treating them as
        equivalent for prev_marker matching would point at a position whose
        actual bytes differ — guaranteed cache miss. Clients MUST be consistent
        about which form they produce (ShanClaw gateway.go normalizes before
        send)."""
        msg_str = {"role": "assistant", "content": "Hello world"}
        msg_block = {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}
        # Different hashes on purpose — see docstring.
        assert _msg_stable_hash(msg_str) != _msg_stable_hash(msg_block)

    def test_tool_use_input_key_ordering(self):
        """Go map iteration order is randomized; the same logical tool_use call
        can serialize with different key orders per goroutine. Hash must absorb."""
        msg1 = {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "file_read",
             "input": {"path": "/a", "line": 5}}
        ]}
        msg2 = {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "file_read",
             "input": {"line": 5, "path": "/a"}}
        ]}
        assert _msg_stable_hash(msg1) == _msg_stable_hash(msg2)

    def test_tool_use_nested_input_key_ordering(self):
        """Nested map keys too — covers tool inputs with structured parameters."""
        msg1 = {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "bash",
             "input": {"cmd": "ls", "opts": {"recursive": True, "color": "auto"}}}
        ]}
        msg2 = {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "bash",
             "input": {"opts": {"color": "auto", "recursive": True}, "cmd": "ls"}}
        ]}
        assert _msg_stable_hash(msg1) == _msg_stable_hash(msg2)

    def test_tool_result_content_shape_is_distinguished(self):
        """Same as test_string_content_shape_is_distinguished: clients must
        pick one form and stick with it. Shannon's gateway sends tool_result as
        plain string, ShanClaw is the source of truth."""
        msg_str = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "result text"}
        ]}
        msg_blk = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": [{"type": "text", "text": "result text"}]}
        ]}
        # Intentionally different — see docstring.
        assert _msg_stable_hash(msg_str) != _msg_stable_hash(msg_blk)

    def test_cache_control_stripped_before_hashing(self):
        """Mutated cache_control markers MUST be transparent to the hash;
        otherwise applying a rolling marker would invalidate its own lookup."""
        msg_clean = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        msg_marked = {"role": "user", "content": [
            {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        ]}
        assert _msg_stable_hash(msg_clean) == _msg_stable_hash(msg_marked)

    def test_cache_control_stripped_nested(self):
        """Nested cache_control at any block depth is stripped."""
        obj = {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "bye"},
            ]
        }
        stripped = _strip_cache_control_for_hash(obj)
        for block in stripped["content"]:
            assert "cache_control" not in block

    def test_tool_use_id_participates_in_hash(self):
        """tool_use IDs are cross-request stable identifiers and MUST affect
        the hash — different IDs signify different prior tool invocations."""
        msg1 = {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "x", "input": {}}
        ]}
        msg2 = {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t2", "name": "x", "input": {}}
        ]}
        assert _msg_stable_hash(msg1) != _msg_stable_hash(msg2)

    def test_role_difference_breaks_hash(self):
        """A user message and an assistant message with identical content must
        hash differently — role is a semantic distinction."""
        msg_user = {"role": "user", "content": "hello"}
        msg_asst = {"role": "assistant", "content": "hello"}
        assert _msg_stable_hash(msg_user) != _msg_stable_hash(msg_asst)

    def test_repeated_hash_is_deterministic(self):
        """Sanity: hashing the same message twice must yield the same digest
        within a process. Covers accidental dependency on iteration order."""
        msg = {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "file_read",
             "input": {"path": "/a", "line": 5, "offset": 10}}
        ]}
        h1 = _msg_stable_hash(msg)
        h2 = _msg_stable_hash(msg)
        assert h1 == h2 and h1 != ""

    def test_non_ascii_content_stable(self):
        """Unicode in content must not trip ensure_ascii or encoding drift.
        Agents on this project do speak Chinese/Japanese frequently."""
        msg1 = {"role": "user", "content": "你好，世界"}
        msg2 = {"role": "user", "content": "你好，世界"}
        assert _msg_stable_hash(msg1) == _msg_stable_hash(msg2) != ""


class TestStripCacheControl:
    """Unit coverage of the hash-input preprocessor."""

    def test_strip_leaves_non_dict_values_unchanged(self):
        assert _strip_cache_control_for_hash("just a string") == "just a string"
        assert _strip_cache_control_for_hash(42) == 42
        assert _strip_cache_control_for_hash(None) is None

    def test_strip_handles_list_of_blocks(self):
        blocks = [
            {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "b"},
        ]
        out = _strip_cache_control_for_hash(blocks)
        assert out == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]

    def test_strip_preserves_non_cache_control_keys(self):
        blk = {"type": "text", "text": "hi", "citations": [1, 2], "cache_control": {}}
        out = _strip_cache_control_for_hash(blk)
        assert out == {"type": "text", "text": "hi", "citations": [1, 2]}
