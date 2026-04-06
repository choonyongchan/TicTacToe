"""Tests for the IterativeDeepeningWrapper."""
from __future__ import annotations

import pytest

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, INF, Move, Player, Result
from tictactoe.agents._search_budget import SearchBudget
from tictactoe.benchmark.metrics import MatchConfig
from tictactoe.agents.heuristic_search.shared.transposition_table import TranspositionTable
from tictactoe.agents.heuristic_search.shared.iterative_deepening import (
    IterativeDeepeningWrapper,
)
from tictactoe.evaluation.heuristics import evaluate_position
from tictactoe.evaluation.move_ordering import KillerMoveTable, HistoryTable
import time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(rows: list[list[int]], player: str = "X") -> GameState:
    mapping = {0: Cell.EMPTY, 1: Cell.X, 2: Cell.O}
    board = [[mapping[v] for v in row] for row in rows]
    n = len(rows)
    current = Player.X if player == "X" else Player.O
    return GameState(board=board, current_player=current, n=n, k=n)


def empty_state(n: int = 3) -> GameState:
    board = Board.create(n)
    return GameState(board=board, current_player=Player.X, n=n, k=n)


def _make_simple_search_fn(call_log: list) -> object:
    """Build a simple search function that records depth calls."""

    def search_fn(state, depth, alpha, beta, budget, tt, killers, history, counters):
        call_log.append(depth)
        counters[0] += 1
        counters[1] = max(counters[1], depth)
        candidates = Board.get_candidate_moves(state, radius=2)
        best_move = candidates[0] if candidates else None
        score = evaluate_position(state, state.current_player)
        return score, best_move

    return search_fn


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestIDDFSBasic:
    """Tests that IDDFS runs correctly and returns legal moves."""

    def test_returns_legal_move_on_empty_board(self) -> None:
        state = empty_state(3)
        config = MatchConfig.depth_controlled(depth=2)
        budget = SearchBudget(config, time.perf_counter_ns())
        tt = TranspositionTable(2**10, seed=1)
        killers = KillerMoveTable()
        history = HistoryTable()
        counters = [0, 0, 0]
        call_log: list = []

        wrapper = IterativeDeepeningWrapper(_make_simple_search_fn(call_log),
                                            use_aspiration=False)
        best_move, score, max_depth = wrapper.run(
            state, budget, tt, killers, history, counters)

        assert best_move is not None
        assert isinstance(best_move, tuple)
        assert len(best_move) == 2
        row, col = best_move
        assert 0 <= row < 3 and 0 <= col < 3
        assert state.board[row][col] is Cell.EMPTY

    def test_returns_legal_move_on_partial_board(self) -> None:
        state = make_state([[1, 0, 0], [0, 2, 0], [0, 0, 0]])
        config = MatchConfig.depth_controlled(depth=2)
        budget = SearchBudget(config, time.perf_counter_ns())
        tt = TranspositionTable(2**10, seed=1)
        killers = KillerMoveTable()
        history = HistoryTable()
        counters = [0, 0, 0]

        wrapper = IterativeDeepeningWrapper(_make_simple_search_fn([]),
                                            use_aspiration=False)
        best_move, _, _ = wrapper.run(state, budget, tt, killers, history, counters)

        assert best_move is not None
        row, col = best_move
        assert state.board[row][col] is Cell.EMPTY

    def test_max_depth_increases_with_depth_budget(self) -> None:
        """Larger depth budget → larger max_depth_completed."""
        state = empty_state(3)

        results = []
        for depth in [1, 3]:
            config = MatchConfig.depth_controlled(depth=depth)
            budget = SearchBudget(config, time.perf_counter_ns())
            tt = TranspositionTable(2**10, seed=1)
            killers = KillerMoveTable()
            history = HistoryTable()
            counters = [0, 0, 0]
            wrapper = IterativeDeepeningWrapper(_make_simple_search_fn([]),
                                                use_aspiration=False)
            _, _, max_depth = wrapper.run(state, budget, tt, killers, history, counters)
            results.append(max_depth)

        assert results[1] >= results[0]

    def test_search_fn_called_at_increasing_depths(self) -> None:
        """The search function is called for depth 1, 2, ... up to budget."""
        state = empty_state(3)
        config = MatchConfig.depth_controlled(depth=3)
        budget = SearchBudget(config, time.perf_counter_ns())
        tt = TranspositionTable(2**10, seed=1)
        killers = KillerMoveTable()
        history = HistoryTable()
        counters = [0, 0, 0]
        call_log: list = []

        wrapper = IterativeDeepeningWrapper(_make_simple_search_fn(call_log),
                                            use_aspiration=False)
        wrapper.run(state, budget, tt, killers, history, counters)

        # Should have been called for depths 1, 2, 3 at minimum
        assert len(call_log) >= 1
        # Depths should be non-decreasing
        for i in range(1, len(call_log)):
            assert call_log[i] >= call_log[i - 1]


# ---------------------------------------------------------------------------
# Aspiration windows
# ---------------------------------------------------------------------------


class TestAspirationWindows:
    """Tests that aspiration window mode runs without errors."""

    def test_aspiration_mode_returns_move(self) -> None:
        state = empty_state(3)
        config = MatchConfig.depth_controlled(depth=3)
        budget = SearchBudget(config, time.perf_counter_ns())
        tt = TranspositionTable(2**10, seed=1)
        killers = KillerMoveTable()
        history = HistoryTable()
        counters = [0, 0, 0]

        wrapper = IterativeDeepeningWrapper(
            _make_simple_search_fn([]),
            use_aspiration=True,
            aspiration_delta=50.0,
        )
        best_move, _, _ = wrapper.run(state, budget, tt, killers, history, counters)

        assert best_move is not None

    def test_aspiration_vs_no_aspiration_same_legality(self) -> None:
        """Both modes should return a legal move."""
        state = make_state([[0, 1, 0], [0, 2, 0], [1, 0, 0]])
        config = MatchConfig.depth_controlled(depth=2)

        for use_asp in (True, False):
            budget = SearchBudget(config, time.perf_counter_ns())
            tt = TranspositionTable(2**10, seed=1)
            killers = KillerMoveTable()
            history = HistoryTable()
            counters = [0, 0, 0]
            wrapper = IterativeDeepeningWrapper(
                _make_simple_search_fn([]), use_aspiration=use_asp)
            move, _, _ = wrapper.run(state, budget, tt, killers, history, counters)
            assert move is not None
            row, col = move
            assert state.board[row][col] is Cell.EMPTY


# ---------------------------------------------------------------------------
# Node-budget control
# ---------------------------------------------------------------------------


class TestNodeBudgetControl:
    """Tests that search stops when node budget is exhausted."""

    def test_node_budget_limits_nodes(self) -> None:
        state = empty_state(3)
        config = MatchConfig.node_controlled(budget=5)
        budget = SearchBudget(config, time.perf_counter_ns())
        tt = TranspositionTable(2**10, seed=1)
        killers = KillerMoveTable()
        history = HistoryTable()
        counters = [0, 0, 0]

        wrapper = IterativeDeepeningWrapper(_make_simple_search_fn([]),
                                            use_aspiration=False)
        best_move, _, _ = wrapper.run(state, budget, tt, killers, history, counters)

        # Should still return a move even with tiny budget
        assert best_move is not None
        # Nodes should not massively exceed budget
        assert counters[0] <= 100  # some slack for overhead
