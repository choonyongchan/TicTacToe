"""Search budget enforcement for time/node/depth-controlled search.

Used by all Tier 1 and Tier 2 search agents to check whether their
allocated search budget has been exhausted.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import time

from tictactoe.benchmark.metrics import MatchConfig
from tictactoe.config import get_config, ConfigError
from tictactoe.core.types import MatchMode


class SearchBudget:
    """Encapsulates all budget-checking logic for a single move.

    Constructed at the start of each choose_move call with the agent's
    MatchConfig and the wall-clock time at which the search began.
    """

    def __init__(self, match_config: MatchConfig | None, start_time_ns: int) -> None:
        """Initialise the budget from a MatchConfig.

        Args:
            match_config: The budget configuration. If None, defaults to
                TIME_CONTROLLED with a 1000 ms limit.
            start_time_ns: The time.perf_counter_ns() value captured at the
                start of the search.
        """
        self._start_ns = start_time_ns
        if match_config is None:
            self._mode = MatchMode.TIME_CONTROLLED
            self._time_limit_ns = int(1_000 * 1_000_000)  # 1000 ms
            self._node_budget = 100_000
            self._fixed_depth = 100
        else:
            self._mode = match_config.mode
            self._time_limit_ns = int(match_config.time_limit_ms * 1_000_000)
            self._node_budget = match_config.node_budget
            self._fixed_depth = match_config.fixed_depth

    def exhausted(self, nodes: int, depth: int) -> bool:
        """Return True if the search budget has been consumed.

        Checks the appropriate limit based on the MatchMode:
        - TIME_CONTROLLED: wall-clock time since construction.
        - NODE_CONTROLLED: nodes visited so far.
        - DEPTH_CONTROLLED: current search depth vs fixed_depth.

        Args:
            nodes: Total nodes visited so far in this search.
            depth: Current iterative-deepening depth (or depth from root).

        Returns:
            True if the budget is exhausted and the search should stop.
        """
        if self._mode is MatchMode.TIME_CONTROLLED:
            elapsed_ns = time.perf_counter_ns() - self._start_ns
            return elapsed_ns >= self._time_limit_ns
        elif self._mode is MatchMode.NODE_CONTROLLED:
            return nodes >= self._node_budget
        elif self._mode is MatchMode.DEPTH_CONTROLLED:
            return depth > self._fixed_depth
        return False

    def max_depth(self) -> int:
        """Return the effective maximum search depth for iterative deepening.

        Returns:
            fixed_depth for DEPTH_CONTROLLED mode; otherwise id_max_depth from
            config (default 1000). The time/node budget will stop the search
            before this limit in practice. Falls back to 100 only when config
            has not been loaded (e.g. during unit tests that bypass main()).
        """
        if self._mode is MatchMode.DEPTH_CONTROLLED:
            return self._fixed_depth
        try:
            return get_config().search.id_max_depth
        except ConfigError:
            return 100  # test-only fallback

    def mode(self) -> MatchMode:
        """Return the active MatchMode.

        Returns:
            The MatchMode enum value controlling this budget.
        """
        return self._mode
