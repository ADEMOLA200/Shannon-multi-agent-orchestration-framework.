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
