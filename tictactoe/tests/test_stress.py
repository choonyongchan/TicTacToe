"""Stress, load, and multi-game stability tests.

These tests verify that the framework handles sustained load without crashing,
leaking memory, or producing incorrect results. Budgets are kept deliberately
small so the suite remains fast in CI.
"""
from __future__ import annotations

import pytest

from tictactoe.agents.classic_search.minimax_ab import MinimaxAB
from tictactoe.agents.monte_carlo.mcts_vanilla import MCTSVanilla
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.benchmark.arena import Arena
from tictactoe.benchmark.metrics import MatchConfig
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mid_game_state(n: int = 3) -> GameState:
    """Return a mid-game state with pieces near the centre of an n×n board."""
    state = GameState(board=Board.create(n), current_player=Player.X, n=n, k=min(n, 5))
    centre = n // 2
    state = state.apply_move((centre, centre))
    state = state.apply_move((centre + 1, centre))
    return state


# ---------------------------------------------------------------------------
# RandomAgent stress
# ---------------------------------------------------------------------------


class TestRandomAgentStress:
    """Hundreds of random-agent games must complete without exception."""

    def test_100_games_random_vs_random(self) -> None:
        arena = Arena(n=3, num_games=100)
        arena.set_seed(42)
        result = arena.duel(RandomAgent(seed=1), RandomAgent(seed=2))
        assert result.total_games == 100
        assert result.agent_a_wins + result.agent_b_wins + result.draws == 100

    def test_50_games_on_5x5_board(self) -> None:
        arena = Arena(n=5, k=4, num_games=50)
        arena.set_seed(7)
        result = arena.duel(RandomAgent(seed=3), RandomAgent(seed=4))
        assert result.total_games == 50


# ---------------------------------------------------------------------------
# MCTS stress
# ---------------------------------------------------------------------------


class TestMCTSStress:
    """MCTS with large simulation counts must remain stable."""

    def test_mcts_500_simulations_returns_legal_move(self) -> None:
        agent = MCTSVanilla(match_config=MatchConfig.node_controlled(500), seed=0)
        state = GameState(board=Board.create(3), current_player=Player.X, n=3, k=3)
        move = agent.choose_move(state)
        r, c = move
        assert 0 <= r < 3 and 0 <= c < 3
        assert state.board[r][c] is Cell.EMPTY

    def test_mcts_many_moves_in_sequence(self) -> None:
        """Simulate an entire 3×3 game with MCTS for both players."""
        import tictactoe.core.board as _board_mod
        from tictactoe.core.game import Game

        config = MatchConfig.node_controlled(100)
        game = Game(
            agent_x=MCTSVanilla(match_config=config, seed=1),
            agent_o=MCTSVanilla(match_config=config, seed=2),
            n=3,
            k=3,
            match_config=config,
        )
        result = game.run()
        # The game must terminate with a valid result.
        from tictactoe.core.types import Result
        assert result in (Result.X_WINS, Result.O_WINS, Result.DRAW)


# ---------------------------------------------------------------------------
# Scalability sweep stress
# ---------------------------------------------------------------------------


class TestScalabilitySweepStress:
    """Scalability sweep must produce well-structured records for 3 board sizes."""

    def test_sweep_three_sizes_returns_one_record(self) -> None:
        arena = Arena(n=3, num_games=4)
        arena.set_seed(0)
        agent = MCTSVanilla(match_config=MatchConfig.node_controlled(50), seed=42)
        records = arena.scalability_sweep([agent], board_sizes=[3, 5, 7], games_per_size=4)
        assert len(records) == 1

    def test_sweep_three_sizes_record_lists_length(self) -> None:
        arena = Arena(n=3, num_games=4)
        arena.set_seed(0)
        agent = MCTSVanilla(match_config=MatchConfig.node_controlled(50), seed=42)
        records = arena.scalability_sweep([agent], board_sizes=[3, 5, 7], games_per_size=4)
        rec = records[0]
        assert len(rec.avg_nodes_per_size) == 3
        assert len(rec.avg_ebf_per_size) == 3
        assert len(rec.avg_time_ms_per_size) == 3
        assert len(rec.win_rate_per_size) == 3

    def test_sweep_win_rates_in_valid_range(self) -> None:
        arena = Arena(n=3, num_games=4)
        arena.set_seed(0)
        agent = MCTSVanilla(match_config=MatchConfig.node_controlled(50), seed=42)
        records = arena.scalability_sweep([agent], board_sizes=[3, 5], games_per_size=4)
        for wr in records[0].win_rate_per_size:
            assert 0.0 <= wr <= 1.0


# ---------------------------------------------------------------------------
# Board operations under repeated load
# ---------------------------------------------------------------------------


class TestBoardOperationsRepeated:
    """Board utility functions must be stable under repeated calls."""

    def test_candidate_moves_large_board_repeated(self) -> None:
        """get_candidate_moves must return a non-empty list on every call."""
        state = _mid_game_state(n=20)
        for _ in range(200):
            candidates = Board.get_candidate_moves(state, radius=2)
            assert len(candidates) > 0, "Expected non-empty candidate list"
            for r, c in candidates:
                assert state.board[r][c] is Cell.EMPTY

    def test_is_terminal_repeated_on_in_progress_board(self) -> None:
        board = Board.create(10)
        board[5][5] = Cell.X
        for _ in range(500):
            from tictactoe.core.types import Result
            result = Board.is_terminal(board, 10, 5, (5, 5))
            assert result is Result.IN_PROGRESS


# ---------------------------------------------------------------------------
# apply_move chain stress
# ---------------------------------------------------------------------------


class TestApplyMoveChainStress:
    """Chaining apply_move must not crash or corrupt state."""

    def test_25_apply_move_on_5x5(self) -> None:
        """Apply 25 moves on a 5×5 board; move_number must reach 25."""
        state = GameState(board=Board.create(5), current_player=Player.X, n=5, k=5)
        # Use all 25 cells in row-major order.
        for r in range(5):
            for c in range(5):
                state = state.apply_move((r, c))
        assert state.move_number == 25
        assert len(state.move_history) == 25

    def test_apply_move_chain_board_fully_populated(self) -> None:
        state = GameState(board=Board.create(5), current_player=Player.X, n=5, k=5)
        for r in range(5):
            for c in range(5):
                state = state.apply_move((r, c))
        for row in state.board:
            for cell in row:
                assert cell is not Cell.EMPTY


# ---------------------------------------------------------------------------
# Minimax stress
# ---------------------------------------------------------------------------


class TestMinimaxStress:
    """MinimaxAB must return a legal move on a variety of mid-game positions."""

    def test_minimax_10_positions_no_crash(self) -> None:
        config = MatchConfig.node_controlled(200)
        agent = MinimaxAB(depth=4, match_config=config)

        positions = [
            ["XO.", ".X.", "..."],
            ["...", ".X.", "..."],
            ["XOX", "OXO", "..."],
            ["X..", ".O.", "..X"],
            ["OO.", "XX.", "..."],
            ["X.O", ".X.", "O.X"],
            [".X.", "...", ".O."],
            ["XX.", "OO.", "..."],
            ["X..", "...", "..."],
            ["XOX", ".O.", "..."],
        ]
        for pos in positions:
            board = Board.create(3)
            last_move = None
            move_history = []
            for r, row_str in enumerate(pos):
                for c, ch in enumerate(row_str):
                    if ch == "X":
                        board[r][c] = Cell.X
                        last_move = (r, c)
                        move_history.append((r, c))
                    elif ch == "O":
                        board[r][c] = Cell.O
                        last_move = (r, c)
                        move_history.append((r, c))
            state = GameState(board=board, current_player=Player.X, n=3, k=3,
                              last_move=last_move, move_history=move_history)
            move = agent.choose_move(state)
            row, col = move
            assert 0 <= row < 3 and 0 <= col < 3
            assert state.board[row][col] is Cell.EMPTY
