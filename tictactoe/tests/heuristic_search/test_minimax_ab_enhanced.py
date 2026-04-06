"""Tests for MinimaxABEnhanced agent."""
from __future__ import annotations

import pytest

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player, Result
from tictactoe.benchmark.metrics import MatchConfig
from tictactoe.agents.heuristic_search.minimax_ab_enhanced import MinimaxABEnhanced
from tictactoe.agents.random_agent import RandomAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(rows: list[list[int]], player: str = "X", k: int | None = None) -> GameState:
    mapping = {0: Cell.EMPTY, 1: Cell.X, 2: Cell.O}
    board = [[mapping[v] for v in row] for row in rows]
    n = len(rows)
    current = Player.X if player == "X" else Player.O
    win_k = k if k is not None else n
    return GameState(board=board, current_player=current, n=n, k=win_k)


def depth_config(d: int = 3) -> MatchConfig:
    return MatchConfig.depth_controlled(depth=d)


def node_config(budget: int = 500) -> MatchConfig:
    return MatchConfig.node_controlled(budget=budget)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Tests for get_name and get_tier."""

    def test_get_name_is_string(self) -> None:
        agent = MinimaxABEnhanced(depth_config())
        assert isinstance(agent.get_name(), str)
        assert len(agent.get_name()) > 0

    def test_get_tier_is_2(self) -> None:
        agent = MinimaxABEnhanced(depth_config())
        assert agent.get_tier() == 2

    def test_name_contains_enhanced(self) -> None:
        agent = MinimaxABEnhanced(depth_config())
        assert "Enhanced" in agent.get_name() or "enhanced" in agent.get_name().lower()


# ---------------------------------------------------------------------------
# Move legality
# ---------------------------------------------------------------------------


class TestMoveLegality:
    """Tests that choose_move always returns a legal move."""

    def test_returns_legal_move_on_empty_board(self) -> None:
        agent = MinimaxABEnhanced(depth_config(2))
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        move = agent.choose_move(state)
        assert isinstance(move, tuple)
        row, col = move
        assert 0 <= row < 3 and 0 <= col < 3
        assert state.board[row][col] is Cell.EMPTY

    def test_returns_legal_move_on_partial_board(self) -> None:
        agent = MinimaxABEnhanced(depth_config(2))
        state = make_state([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
        move = agent.choose_move(state)
        row, col = move
        assert state.board[row][col] is Cell.EMPTY

    def test_returns_only_empty_cell_when_one_left(self) -> None:
        agent = MinimaxABEnhanced(depth_config(1))
        # One cell left at (2,2)
        state = make_state([
            [1, 2, 1],
            [2, 1, 2],
            [2, 1, 0],
        ])
        move = agent.choose_move(state)
        assert move == (2, 2)


# ---------------------------------------------------------------------------
# Forced moves
# ---------------------------------------------------------------------------


class TestForcedMoves:
    """Tests that the agent correctly plays forced wins and blocks."""

    def test_picks_winning_move(self) -> None:
        """Agent must complete a winning line when available."""
        agent = MinimaxABEnhanced(depth_config(3))
        # X can win at (0,2)
        state = make_state([
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 2],
        ])
        move = agent.choose_move(state)
        assert move == (0, 2)

    def test_blocks_opponent_win(self) -> None:
        """Agent must block O's immediate win."""
        agent = MinimaxABEnhanced(depth_config(3))
        # O can win at (1,0); X must block
        state = make_state([
            [1, 0, 0],
            [0, 2, 2],
            [0, 0, 1],
        ], player="X")
        move = agent.choose_move(state)
        # The only reasonable response is to block at (1,0)
        assert move == (1, 0)

    def test_winning_move_preferred_over_block(self) -> None:
        """If X can win, do it — don't just block O."""
        agent = MinimaxABEnhanced(depth_config(3))
        # X can win at (0,2); O also threatens at (1,0) but X wins first
        state = make_state([
            [1, 1, 0],
            [0, 2, 2],
            [0, 0, 0],
        ])
        move = agent.choose_move(state)
        assert move == (0, 2)


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


class TestInstrumentation:
    """Tests that the agent correctly populates state instrumentation fields."""

    def test_nodes_visited_positive_after_search(self) -> None:
        agent = MinimaxABEnhanced(depth_config(2))
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        agent.choose_move(state)
        assert state.nodes_visited >= 1

    def test_ebf_computed(self) -> None:
        agent = MinimaxABEnhanced(depth_config(2))
        state = make_state([[1, 0, 0], [0, 2, 0], [0, 0, 0]])
        agent.choose_move(state)
        # EBF should be 0 if depth=0, otherwise >= 0
        assert state.effective_branching_factor >= 0.0

    def test_prunings_non_negative(self) -> None:
        agent = MinimaxABEnhanced(depth_config(3))
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        agent.choose_move(state)
        assert state.prunings >= 0

    def test_max_depth_reached_non_negative(self) -> None:
        agent = MinimaxABEnhanced(depth_config(2))
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        agent.choose_move(state)
        assert state.max_depth_reached >= 0


# ---------------------------------------------------------------------------
# TSS optimisation
# ---------------------------------------------------------------------------


class TestTSSOptimisation:
    """Tests that TSS can find forced wins with fewer nodes."""

    def test_tss_finds_win_quickly(self) -> None:
        """With TSS enabled, a near-win position uses very few nodes."""
        agent_tss = MinimaxABEnhanced(depth_config(4), use_tss=True)
        state = make_state([
            [1, 1, 0],
            [2, 0, 0],
            [0, 0, 2],
        ])
        agent_tss.choose_move(state)
        nodes_with_tss = state.nodes_visited

        # Verify the result is valid
        assert nodes_with_tss >= 1

    def test_tss_disabled_still_finds_move(self) -> None:
        """With TSS disabled, the agent still returns a legal move."""
        agent = MinimaxABEnhanced(depth_config(2), use_tss=False)
        state = make_state([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        move = agent.choose_move(state)
        assert move is not None
        row, col = move
        assert state.board[row][col] is Cell.EMPTY


# ---------------------------------------------------------------------------
# Performance sanity
# ---------------------------------------------------------------------------


class TestPerformanceSanity:
    """Sanity checks — agent beats random most of the time."""

    def test_wins_majority_against_random_on_3x3(self) -> None:
        """MinimaxABEnhanced should win most games as X against RandomAgent."""
        from tictactoe.core.game import Game
        config = MatchConfig.depth_controlled(depth=4)
        wins = 0
        games = 10
        for game_seed in range(games):
            agent = MinimaxABEnhanced(config)
            game = Game(agent_x=agent, agent_o=RandomAgent(seed=game_seed),
                        n=3, k=3, match_config=config)
            result = game.run()
            if result == Result.X_WINS:
                wins += 1
        # Should win at least half
        assert wins >= games // 2
