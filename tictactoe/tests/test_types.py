"""Tests for tictactoe/core/types.py.

Verifies enum methods, enum distinctness, and constant values. These are
foundational contracts that every other module relies on.
"""
from __future__ import annotations

import math

import pytest

from tictactoe.core.types import (
    INF,
    DEFAULT_NODE_BUDGET,
    DEFAULT_TIME_LIMIT_MS,
    RANDOM_AGENT_WIN_THRESHOLD,
    WIN_SCORE,
    Cell,
    MatchMode,
    Player,
    Result,
)


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------


class TestPlayer:
    """Tests for the Player enum and its methods."""

    def test_x_opponent_is_o(self) -> None:
        assert Player.X.opponent() is Player.O

    def test_o_opponent_is_x(self) -> None:
        assert Player.O.opponent() is Player.X

    def test_opponent_is_involution_for_x(self) -> None:
        """Applying opponent() twice must return the original player."""
        assert Player.X.opponent().opponent() is Player.X

    def test_opponent_is_involution_for_o(self) -> None:
        assert Player.O.opponent().opponent() is Player.O

    def test_x_to_cell_is_cell_x(self) -> None:
        assert Player.X.to_cell() is Cell.X

    def test_o_to_cell_is_cell_o(self) -> None:
        assert Player.O.to_cell() is Cell.O

    def test_to_cell_not_empty(self) -> None:
        assert Player.X.to_cell() is not Cell.EMPTY
        assert Player.O.to_cell() is not Cell.EMPTY


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------


class TestCell:
    """Tests for the Cell enum."""

    def test_empty_x_o_are_distinct(self) -> None:
        assert Cell.EMPTY is not Cell.X
        assert Cell.EMPTY is not Cell.O
        assert Cell.X is not Cell.O

    def test_all_three_values_exist(self) -> None:
        assert len({Cell.EMPTY, Cell.X, Cell.O}) == 3


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class TestResult:
    """Tests for the Result enum."""

    def test_four_distinct_values(self) -> None:
        results = {Result.IN_PROGRESS, Result.X_WINS, Result.O_WINS, Result.DRAW}
        assert len(results) == 4

    def test_result_names(self) -> None:
        assert Result.IN_PROGRESS.name == "IN_PROGRESS"
        assert Result.X_WINS.name == "X_WINS"
        assert Result.O_WINS.name == "O_WINS"
        assert Result.DRAW.name == "DRAW"

    def test_in_progress_is_not_terminal(self) -> None:
        """Sanity: IN_PROGRESS is not equal to any terminal result."""
        assert Result.IN_PROGRESS is not Result.X_WINS
        assert Result.IN_PROGRESS is not Result.O_WINS
        assert Result.IN_PROGRESS is not Result.DRAW


# ---------------------------------------------------------------------------
# MatchMode
# ---------------------------------------------------------------------------


class TestMatchMode:
    """Tests for the MatchMode enum."""

    def test_three_distinct_values(self) -> None:
        modes = {MatchMode.TIME_CONTROLLED, MatchMode.NODE_CONTROLLED, MatchMode.DEPTH_CONTROLLED}
        assert len(modes) == 3

    def test_mode_names(self) -> None:
        assert MatchMode.TIME_CONTROLLED.name == "TIME_CONTROLLED"
        assert MatchMode.NODE_CONTROLLED.name == "NODE_CONTROLLED"
        assert MatchMode.DEPTH_CONTROLLED.name == "DEPTH_CONTROLLED"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level numeric constants."""

    def test_inf_is_positive_infinity(self) -> None:
        assert math.isinf(INF)
        assert INF > 0

    def test_win_score_is_one_million(self) -> None:
        assert WIN_SCORE == 1_000_000.0

    def test_win_score_is_finite(self) -> None:
        """WIN_SCORE must be a finite float so it can be compared in search."""
        assert math.isfinite(WIN_SCORE)

    def test_default_time_limit_ms_is_1000(self) -> None:
        assert DEFAULT_TIME_LIMIT_MS == 1_000.0

    def test_default_node_budget_is_100k(self) -> None:
        assert DEFAULT_NODE_BUDGET == 100_000

    def test_random_agent_win_threshold(self) -> None:
        assert RANDOM_AGENT_WIN_THRESHOLD == 0.85

    def test_win_score_less_than_inf(self) -> None:
        """WIN_SCORE must be dominated by INF so search bounds work correctly."""
        assert WIN_SCORE < INF
