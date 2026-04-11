"""Tests for the SearchBudget class.

Covers all three budget modes (TIME_CONTROLLED, NODE_CONTROLLED,
DEPTH_CONTROLLED), the None-config fallback, max_depth(), and mode().
"""
from __future__ import annotations

import time

import pytest

from tictactoe.agents._search_budget import SearchBudget
from tictactoe.benchmark.metrics import MatchConfig
from tictactoe.core.types import MatchMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _budget(match_config: MatchConfig | None) -> SearchBudget:
    """Construct a SearchBudget with the current time as start."""
    return SearchBudget(match_config, time.perf_counter_ns())


# ---------------------------------------------------------------------------
# TIME_CONTROLLED
# ---------------------------------------------------------------------------


class TestTimeControlled:
    """Tests for TIME_CONTROLLED budget mode."""

    def test_not_exhausted_immediately(self) -> None:
        """A fresh time budget must not report exhausted right away."""
        budget = _budget(MatchConfig.time_controlled(ms=1000.0))
        assert budget.exhausted(0, 0) is False

    def test_exhausted_after_sleep_past_limit(self) -> None:
        """Budget must report exhausted after sleeping beyond the time limit."""
        budget = _budget(MatchConfig.time_controlled(ms=50.0))
        time.sleep(0.1)  # 100 ms > 50 ms limit
        assert budget.exhausted(0, 0) is True

    def test_zero_ms_limit_exhausted_immediately(self) -> None:
        """A 0 ms time limit should be exhausted on the very first check."""
        budget = _budget(MatchConfig.time_controlled(ms=0.0))
        # Even without a sleep, 0 ms will always be exceeded.
        time.sleep(0.001)  # tiny delay to ensure non-zero elapsed
        assert budget.exhausted(0, 0) is True

    def test_node_and_depth_args_ignored_in_time_mode(self) -> None:
        """Node and depth arguments must have no effect in TIME_CONTROLLED mode."""
        budget = _budget(MatchConfig.time_controlled(ms=2000.0))
        # Large node/depth values should not trigger exhaustion.
        assert budget.exhausted(10_000_000, 9999) is False


# ---------------------------------------------------------------------------
# NODE_CONTROLLED
# ---------------------------------------------------------------------------


class TestNodeControlled:
    """Tests for NODE_CONTROLLED budget mode."""

    def test_not_exhausted_below_budget(self) -> None:
        budget = _budget(MatchConfig.node_controlled(100))
        assert budget.exhausted(99, 0) is False

    def test_exhausted_at_budget(self) -> None:
        budget = _budget(MatchConfig.node_controlled(100))
        assert budget.exhausted(100, 0) is True

    def test_exhausted_above_budget(self) -> None:
        budget = _budget(MatchConfig.node_controlled(100))
        assert budget.exhausted(110, 0) is True

    def test_budget_zero_exhausted_at_zero_nodes(self) -> None:
        """Zero-node budget exhausted even before the first node is visited."""
        budget = _budget(MatchConfig.node_controlled(0))
        assert budget.exhausted(0, 0) is True

    def test_budget_one_not_exhausted_at_zero(self) -> None:
        budget = _budget(MatchConfig.node_controlled(1))
        assert budget.exhausted(0, 0) is False

    def test_budget_one_exhausted_at_one(self) -> None:
        budget = _budget(MatchConfig.node_controlled(1))
        assert budget.exhausted(1, 0) is True

    def test_depth_arg_ignored_in_node_mode(self) -> None:
        """Depth argument must have no effect in NODE_CONTROLLED mode."""
        budget = _budget(MatchConfig.node_controlled(50))
        assert budget.exhausted(49, 9999) is False
        assert budget.exhausted(50, 0) is True


# ---------------------------------------------------------------------------
# DEPTH_CONTROLLED
# ---------------------------------------------------------------------------


class TestDepthControlled:
    """Tests for DEPTH_CONTROLLED budget mode.

    The condition is ``depth > fixed_depth``, so the budget is NOT exhausted
    when depth equals fixed_depth — only when it strictly exceeds it.
    """

    def test_not_exhausted_at_fixed_depth(self) -> None:
        """depth == fixed_depth should NOT be exhausted."""
        budget = _budget(MatchConfig.depth_controlled(4))
        assert budget.exhausted(0, 4) is False

    def test_exhausted_one_above_fixed_depth(self) -> None:
        """depth == fixed_depth + 1 must be exhausted."""
        budget = _budget(MatchConfig.depth_controlled(4))
        assert budget.exhausted(0, 5) is True

    def test_not_exhausted_at_zero(self) -> None:
        budget = _budget(MatchConfig.depth_controlled(4))
        assert budget.exhausted(0, 0) is False

    def test_not_exhausted_at_depth_minus_one(self) -> None:
        budget = _budget(MatchConfig.depth_controlled(10))
        assert budget.exhausted(999_999, 9) is False

    def test_nodes_arg_ignored_in_depth_mode(self) -> None:
        """Node count must have no effect in DEPTH_CONTROLLED mode."""
        budget = _budget(MatchConfig.depth_controlled(3))
        assert budget.exhausted(999_999, 3) is False  # at limit, not beyond
        assert budget.exhausted(0, 4) is True          # beyond limit


# ---------------------------------------------------------------------------
# None match_config (defaults)
# ---------------------------------------------------------------------------


class TestNoneMatchConfig:
    """Tests for the fallback behaviour when match_config is None."""

    def test_defaults_to_time_controlled(self) -> None:
        """None config must default to TIME_CONTROLLED mode."""
        budget = SearchBudget(None, time.perf_counter_ns())
        assert budget.mode() is MatchMode.TIME_CONTROLLED

    def test_not_exhausted_immediately_with_none_config(self) -> None:
        """The default 1000 ms budget must not be exhausted right away."""
        budget = SearchBudget(None, time.perf_counter_ns())
        assert budget.exhausted(0, 0) is False


# ---------------------------------------------------------------------------
# max_depth
# ---------------------------------------------------------------------------


class TestMaxDepth:
    """Tests for SearchBudget.max_depth()."""

    def test_depth_controlled_returns_fixed_depth(self) -> None:
        budget = _budget(MatchConfig.depth_controlled(7))
        assert budget.max_depth() == 7

    def test_time_controlled_returns_fallback(self) -> None:
        """With no config loaded, max_depth falls back to 100."""
        budget = _budget(MatchConfig.time_controlled(500.0))
        # Tests bypass main() so no config is loaded; fallback is 100.
        assert budget.max_depth() == 100

    def test_node_controlled_returns_fallback(self) -> None:
        budget = _budget(MatchConfig.node_controlled(1000))
        assert budget.max_depth() == 100


# ---------------------------------------------------------------------------
# mode
# ---------------------------------------------------------------------------


class TestMode:
    """Tests for SearchBudget.mode()."""

    def test_time_controlled_mode(self) -> None:
        assert _budget(MatchConfig.time_controlled()).mode() is MatchMode.TIME_CONTROLLED

    def test_node_controlled_mode(self) -> None:
        assert _budget(MatchConfig.node_controlled()).mode() is MatchMode.NODE_CONTROLLED

    def test_depth_controlled_mode(self) -> None:
        assert _budget(MatchConfig.depth_controlled()).mode() is MatchMode.DEPTH_CONTROLLED
