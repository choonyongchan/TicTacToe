"""Tests for the NegaScout (PVS) classic search agent."""
from __future__ import annotations

import pytest

from tictactoe.agents.classic_search.negascout import NegaScout
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player, Result
from tictactoe.benchmark.metrics import MatchConfig
from tictactoe.benchmark.arena import Arena
from tictactoe.benchmark.correctness import BruteForceOracle


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_state(board_rows: list[str], player: Player = Player.X, k: int | None = None) -> GameState:
    """Create GameState from list of strings like ['XX.', '...', '...']."""
    n = len(board_rows)
    if k is None:
        k = n
    board = Board.create(n)
    for r, row_str in enumerate(board_rows):
        for c, ch in enumerate(row_str):
            if ch == 'X':
                board[r][c] = Cell.X
            elif ch == 'O':
                board[r][c] = Cell.O
    last_move = None
    for r in range(n):
        for c in range(n):
            if board[r][c] is not Cell.EMPTY:
                last_move = (r, c)
    return GameState(board=board, current_player=player, n=n, k=k, last_move=last_move)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNegaScoutNeverLoses:
    def test_never_loses_to_oracle_n3(self) -> None:
        """NegaScout should never lose to BruteForceOracle on 3x3."""
        agent = NegaScout(depth=4)
        oracle = BruteForceOracle()
        result = Arena(n=3, num_games=20, match_config=MatchConfig.depth_controlled(4)).duel(agent, oracle)
        assert result.agent_b_wins == 0, (
            f"NegaScout lost {result.agent_b_wins} games to BruteForceOracle"
        )


class TestNegaScoutForcedMoves:
    def test_picks_immediate_winning_move(self) -> None:
        """Agent must complete a winning line when available."""
        agent = NegaScout(depth=4)
        state = make_state(['XX.', '...', '...'], player=Player.X, k=3)
        move = agent.choose_move(state)
        assert move == (0, 2), f"Expected (0,2) but got {move}"

    def test_blocks_immediate_opponent_win(self) -> None:
        """Agent must block the opponent's immediate win."""
        agent = NegaScout(depth=4)
        state = make_state(['OO.', 'X..', 'X..'], player=Player.X, k=3)
        move = agent.choose_move(state)
        assert move == (0, 2), f"Expected (0,2) but got {move}"


class TestNegaScoutLegality:
    def test_returns_legal_move(self) -> None:
        """Chosen move must be in-bounds and on an empty cell."""
        agent = NegaScout(depth=4)
        state = make_state(['XO.', '.X.', '...'], player=Player.X, k=3)
        move = agent.choose_move(state)
        row, col = move
        assert 0 <= row < state.n
        assert 0 <= col < state.n
        assert state.board[row][col] is Cell.EMPTY

    def test_never_returns_none(self) -> None:
        """choose_move must never return None."""
        agent = NegaScout(depth=4)
        state = make_state(['XO.', '.X.', '...'], player=Player.X, k=3)
        move = agent.choose_move(state)
        assert move is not None


class TestNegaScoutInstrumentation:
    def test_nodes_visited_positive(self) -> None:
        """nodes_visited must be positive after a search."""
        agent = NegaScout(depth=4)
        state = make_state(['XO.', '.X.', '...'], player=Player.X, k=3)
        agent.choose_move(state)
        assert state.nodes_visited > 0

    def test_max_depth_not_exceeded(self) -> None:
        """max_depth_reached must not exceed the agent's configured depth."""
        agent = NegaScout(depth=4)
        state = make_state(['XO.', '.X.', '...'], player=Player.X, k=3)
        agent.choose_move(state)
        assert state.max_depth_reached <= agent.depth

    def test_prunings_positive_midgame(self) -> None:
        """Alpha-beta pruning should occur on a mid-game position at depth=4."""
        agent = NegaScout(depth=4)
        state = make_state(['XO.', '.X.', '..O'], player=Player.X, k=3)
        agent.choose_move(state)
        assert state.prunings > 0, "Expected prunings > 0 in mid-game at depth=4"


class TestNegaScoutBudget:
    def test_node_budget_1_returns_legal_move(self) -> None:
        """Agent must still return a legal move with a node budget of 1."""
        agent = NegaScout(depth=4, match_config=MatchConfig.node_controlled(1))
        state = make_state(['XO.', '.X.', '...'], player=Player.X, k=3)
        move = agent.choose_move(state)
        row, col = move
        assert 0 <= row < state.n
        assert 0 <= col < state.n
        assert state.board[row][col] is Cell.EMPTY


class TestNegaScoutDeterminism:
    def test_same_state_same_move(self) -> None:
        """NegaScout is deterministic: same state must return same move."""
        agent = NegaScout(depth=4)
        state1 = make_state(['XO.', '.X.', '...'], player=Player.X, k=3)
        state2 = make_state(['XO.', '.X.', '...'], player=Player.X, k=3)
        move1 = agent.choose_move(state1)
        move2 = agent.choose_move(state2)
        assert move1 == move2
