"""Tests for NegaScoutEnhanced agent."""
from __future__ import annotations

import pytest

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player, Result
from tictactoe.benchmark.metrics import MatchConfig
from tictactoe.agents.heuristic_search.negascout_enhanced import NegaScoutEnhanced
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


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_get_name_is_string(self) -> None:
        agent = NegaScoutEnhanced(depth_config())
        assert isinstance(agent.get_name(), str)
        assert len(agent.get_name()) > 0

    def test_get_tier_is_2(self) -> None:
        agent = NegaScoutEnhanced(depth_config())
        assert agent.get_tier() == 2

    def test_name_contains_negascout_or_enhanced(self) -> None:
        agent = NegaScoutEnhanced(depth_config())
        name_lower = agent.get_name().lower()
        assert "negascout" in name_lower or "pvs" in name_lower or "enhanced" in name_lower


# ---------------------------------------------------------------------------
# Move legality
# ---------------------------------------------------------------------------


class TestMoveLegality:
    def test_returns_legal_move_on_empty_board(self) -> None:
        agent = NegaScoutEnhanced(depth_config(2))
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        move = agent.choose_move(state)
        assert isinstance(move, tuple)
        row, col = move
        assert 0 <= row < 3 and 0 <= col < 3
        assert state.board[row][col] is Cell.EMPTY

    def test_returns_legal_move_on_partial_board(self) -> None:
        agent = NegaScoutEnhanced(depth_config(2))
        state = make_state([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
        move = agent.choose_move(state)
        row, col = move
        assert state.board[row][col] is Cell.EMPTY

    def test_returns_only_empty_cell_when_one_left(self) -> None:
        agent = NegaScoutEnhanced(depth_config(1))
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
    def test_picks_winning_move(self) -> None:
        agent = NegaScoutEnhanced(depth_config(3))
        state = make_state([
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 2],
        ])
        move = agent.choose_move(state)
        assert move == (0, 2)

    def test_blocks_opponent_win(self) -> None:
        agent = NegaScoutEnhanced(depth_config(3))
        state = make_state([
            [1, 0, 0],
            [0, 2, 2],
            [0, 0, 1],
        ], player="X")
        move = agent.choose_move(state)
        assert move == (1, 0)

    def test_winning_move_preferred_over_block(self) -> None:
        agent = NegaScoutEnhanced(depth_config(3))
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
    def test_nodes_visited_positive_after_search(self) -> None:
        agent = NegaScoutEnhanced(depth_config(2))
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        agent.choose_move(state)
        assert state.nodes_visited >= 1

    def test_ebf_computed(self) -> None:
        agent = NegaScoutEnhanced(depth_config(2))
        state = make_state([[1, 0, 0], [0, 2, 0], [0, 0, 0]])
        agent.choose_move(state)
        assert state.effective_branching_factor >= 0.0

    def test_prunings_non_negative(self) -> None:
        agent = NegaScoutEnhanced(depth_config(3))
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        agent.choose_move(state)
        assert state.prunings >= 0

    def test_max_depth_reached_non_negative(self) -> None:
        agent = NegaScoutEnhanced(depth_config(2))
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        agent.choose_move(state)
        assert state.max_depth_reached >= 0


# ---------------------------------------------------------------------------
# PVS-specific behaviour
# ---------------------------------------------------------------------------


class TestPVSBehaviour:
    """NegaScout-specific checks."""

    def test_pvs_still_finds_correct_move_vs_plain_negamax(self) -> None:
        """PVS should find the same winning move as plain alpha-beta on easy positions."""
        from tictactoe.agents.heuristic_search.minimax_ab_enhanced import MinimaxABEnhanced

        config = depth_config(3)
        pvs = NegaScoutEnhanced(config, use_tss=False)
        ab = MinimaxABEnhanced(config, use_tss=False)

        state = make_state([
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 2],
        ])
        pvs_move = pvs.choose_move(state)
        ab_move = ab.choose_move(state)

        # Both should play the winning move
        assert pvs_move == (0, 2)
        assert ab_move == (0, 2)

    def test_tss_disabled_still_works(self) -> None:
        agent = NegaScoutEnhanced(depth_config(2), use_tss=False)
        state = make_state([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        move = agent.choose_move(state)
        assert move is not None
        row, col = move
        assert state.board[row][col] is Cell.EMPTY


# ---------------------------------------------------------------------------
# Performance sanity
# ---------------------------------------------------------------------------


class TestPerformanceSanity:
    def test_wins_majority_against_random_on_3x3(self) -> None:
        from tictactoe.core.game import Game
        config = MatchConfig.depth_controlled(depth=4)
        wins = 0
        games = 10
        for game_seed in range(games):
            agent = NegaScoutEnhanced(config)
            game = Game(agent_x=agent, agent_o=RandomAgent(seed=game_seed),
                        n=3, k=3, match_config=config)
            result = game.run()
            if result == Result.X_WINS:
                wins += 1
        assert wins >= games // 2
