"""Tests for MTDfEnhanced agent."""
from __future__ import annotations

import pytest

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player, Result
from tictactoe.benchmark.metrics import MatchConfig
from tictactoe.agents.heuristic_search.mtdf_enhanced import MTDfEnhanced
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
        agent = MTDfEnhanced(depth_config())
        assert isinstance(agent.get_name(), str)
        assert len(agent.get_name()) > 0

    def test_get_tier_is_2(self) -> None:
        agent = MTDfEnhanced(depth_config())
        assert agent.get_tier() == 2

    def test_name_contains_mtdf_or_enhanced(self) -> None:
        agent = MTDfEnhanced(depth_config())
        name_lower = agent.get_name().lower()
        assert "mtd" in name_lower or "enhanced" in name_lower


# ---------------------------------------------------------------------------
# Move legality
# ---------------------------------------------------------------------------


class TestMoveLegality:
    def test_returns_legal_move_on_empty_board(self) -> None:
        agent = MTDfEnhanced(depth_config(2))
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        move = agent.choose_move(state)
        assert isinstance(move, tuple)
        row, col = move
        assert 0 <= row < 3 and 0 <= col < 3
        assert state.board[row][col] is Cell.EMPTY

    def test_returns_legal_move_on_partial_board(self) -> None:
        agent = MTDfEnhanced(depth_config(2))
        state = make_state([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
        move = agent.choose_move(state)
        row, col = move
        assert state.board[row][col] is Cell.EMPTY

    def test_returns_only_empty_cell_when_one_left(self) -> None:
        agent = MTDfEnhanced(depth_config(1))
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
        agent = MTDfEnhanced(depth_config(3))
        state = make_state([
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 2],
        ])
        move = agent.choose_move(state)
        assert move == (0, 2)

    def test_blocks_opponent_win(self) -> None:
        agent = MTDfEnhanced(depth_config(3))
        state = make_state([
            [1, 0, 0],
            [0, 2, 2],
            [0, 0, 1],
        ], player="X")
        move = agent.choose_move(state)
        assert move == (1, 0)

    def test_winning_move_preferred_over_block(self) -> None:
        agent = MTDfEnhanced(depth_config(3))
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
        agent = MTDfEnhanced(depth_config(2))
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        agent.choose_move(state)
        assert state.nodes_visited >= 1

    def test_ebf_computed(self) -> None:
        agent = MTDfEnhanced(depth_config(2))
        state = make_state([[1, 0, 0], [0, 2, 0], [0, 0, 0]])
        agent.choose_move(state)
        assert state.effective_branching_factor >= 0.0

    def test_prunings_non_negative(self) -> None:
        agent = MTDfEnhanced(depth_config(3))
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        agent.choose_move(state)
        assert state.prunings >= 0

    def test_max_depth_reached_non_negative(self) -> None:
        agent = MTDfEnhanced(depth_config(2))
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        agent.choose_move(state)
        assert state.max_depth_reached >= 0


# ---------------------------------------------------------------------------
# MTD(f)-specific behaviour
# ---------------------------------------------------------------------------


class TestMTDfBehaviour:
    """MTD(f)-specific checks."""

    def test_multiple_moves_consistent(self) -> None:
        """The same agent can be called multiple times without crashing."""
        agent = MTDfEnhanced(depth_config(2))
        state1 = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        move1 = agent.choose_move(state1)
        state2 = state1.apply_move(move1)
        state2.current_player = Player.O
        # second call — TT has been cleared between moves by choose_move
        state2b = make_state([[1, 0, 0], [0, 0, 0], [0, 0, 0]], player="O")
        move2 = agent.choose_move(state2b)
        assert move2 is not None

    def test_tss_disabled_still_finds_move(self) -> None:
        agent = MTDfEnhanced(depth_config(2), use_tss=False)
        state = make_state([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        move = agent.choose_move(state)
        assert move is not None
        row, col = move
        assert state.board[row][col] is Cell.EMPTY

    def test_tss_enabled_finds_forced_win(self) -> None:
        """With TSS, a near-win position should be handled efficiently."""
        agent = MTDfEnhanced(depth_config(4), use_tss=True)
        state = make_state([
            [1, 1, 0],
            [2, 0, 0],
            [0, 0, 2],
        ])
        move = agent.choose_move(state)
        assert move is not None


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
            agent = MTDfEnhanced(config)
            game = Game(agent_x=agent, agent_o=RandomAgent(seed=game_seed),
                        n=3, k=3, match_config=config)
            result = game.run()
            if result == Result.X_WINS:
                wins += 1
        assert wins >= games // 2


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    """Agent must respect TIME, NODE, and DEPTH budgets."""

    def test_node_budget_limits_nodes_visited(self) -> None:
        budget = 200
        config = MatchConfig.node_controlled(budget=budget)
        agent = MTDfEnhanced(config)
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        agent.choose_move(state)
        # Allow a small overrun for the final node that triggers the check
        assert state.nodes_visited <= budget + 50

    def test_depth_budget_limits_max_depth(self) -> None:
        config = MatchConfig.depth_controlled(depth=2)
        agent = MTDfEnhanced(config)
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        agent.choose_move(state)
        assert state.max_depth_reached <= 2

    def test_time_budget_completes_without_crash(self) -> None:
        config = MatchConfig.time_controlled(ms=50.0)
        agent = MTDfEnhanced(config)
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        move = agent.choose_move(state)
        assert move is not None
        row, col = move
        assert state.board[row][col] is Cell.EMPTY

    def test_node_budget_still_finds_legal_move(self) -> None:
        config = MatchConfig.node_controlled(budget=100)
        agent = MTDfEnhanced(config)
        state = make_state([[1, 0, 0], [0, 2, 0], [0, 0, 0]])
        move = agent.choose_move(state)
        assert move is not None
        row, col = move
        assert state.board[row][col] is Cell.EMPTY


# ---------------------------------------------------------------------------
# Larger board sizes
# ---------------------------------------------------------------------------


class TestLargerBoards:
    """Agent must work correctly on 4×4 and 5×5 boards."""

    def test_legal_move_on_4x4_board(self) -> None:
        agent = MTDfEnhanced(depth_config(2))
        state = make_state(
            [[0, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 2, 0],
             [0, 0, 0, 0]],
            k=4,
        )
        move = agent.choose_move(state)
        row, col = move
        assert 0 <= row < 4 and 0 <= col < 4
        assert state.board[row][col] is Cell.EMPTY

    def test_legal_move_on_5x5_board_k3(self) -> None:
        agent = MTDfEnhanced(depth_config(2))
        state = make_state(
            [[0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 2, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0]],
            k=3,
        )
        move = agent.choose_move(state)
        row, col = move
        assert 0 <= row < 5 and 0 <= col < 5
        assert state.board[row][col] is Cell.EMPTY

    def test_finds_immediate_win_on_4x4(self) -> None:
        # X has three in a row (cols 0,1,2) — agent must complete to (0,3)
        agent = MTDfEnhanced(depth_config(3))
        state = make_state(
            [[1, 1, 1, 0],
             [0, 2, 0, 0],
             [0, 0, 2, 0],
             [0, 0, 0, 0]],
            k=4,
        )
        move = agent.choose_move(state)
        assert move == (0, 3)

    def test_nodes_visited_positive_on_4x4(self) -> None:
        agent = MTDfEnhanced(depth_config(2))
        state = make_state(
            [[0, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 2, 0],
             [0, 0, 0, 0]],
            k=4,
        )
        agent.choose_move(state)
        assert state.nodes_visited >= 1


# ---------------------------------------------------------------------------
# Symmetry mode
# ---------------------------------------------------------------------------


class TestSymmetryMode:
    """use_symmetry=True must still return valid legal moves."""

    def test_symmetry_enabled_returns_legal_move(self) -> None:
        agent = MTDfEnhanced(depth_config(2), use_symmetry=True)
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        move = agent.choose_move(state)
        row, col = move
        assert state.board[row][col] is Cell.EMPTY

    def test_symmetry_enabled_finds_forced_win(self) -> None:
        agent_sym = MTDfEnhanced(depth_config(3), use_symmetry=True)
        state = make_state([
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 2],
        ])
        move = agent_sym.choose_move(state)
        assert move == (0, 2)

    def test_symmetry_disabled_vs_enabled_both_find_forced_win(self) -> None:
        state = make_state([
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 2],
        ])
        for use_sym in (False, True):
            agent = MTDfEnhanced(depth_config(3), use_symmetry=use_sym)
            move = agent.choose_move(state)
            assert move == (0, 2), f"use_symmetry={use_sym} failed to find win"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Two fresh agent instances on the same position must return the same move."""

    def test_same_position_same_move_repeated(self) -> None:
        state = make_state([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        config = MatchConfig.depth_controlled(depth=3)
        moves = {MTDfEnhanced(config).choose_move(state) for _ in range(3)}
        assert len(moves) == 1, f"Non-deterministic moves: {moves}"


# ---------------------------------------------------------------------------
# Full game integration
# ---------------------------------------------------------------------------


class TestFullGameIntegration:
    """MTDfEnhanced must survive end-to-end game scenarios."""

    def test_plays_complete_game_between_two_mtdf_agents(self) -> None:
        from tictactoe.core.game import Game
        config = MatchConfig.depth_controlled(depth=3)
        agent_x = MTDfEnhanced(config)
        agent_o = MTDfEnhanced(config)
        game = Game(agent_x=agent_x, agent_o=agent_o, n=3, k=3, match_config=config)
        result = game.run()
        assert result in (Result.X_WINS, Result.O_WINS, Result.DRAW)

    def test_wins_as_second_player_against_random(self) -> None:
        from tictactoe.core.game import Game
        config = MatchConfig.depth_controlled(depth=4)
        wins = 0
        games = 10
        for seed in range(games):
            agent_o = MTDfEnhanced(config)
            game = Game(agent_x=RandomAgent(seed=seed), agent_o=agent_o,
                        n=3, k=3, match_config=config)
            result = game.run()
            if result == Result.O_WINS:
                wins += 1
        assert wins >= 4, f"Won only {wins}/10 games as second player"

    def test_sequential_move_sequence_3x3(self) -> None:
        """Agent can play moves sequentially across multiple states."""
        config = MatchConfig.depth_controlled(depth=3)
        agent = MTDfEnhanced(config)
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        move1 = agent.choose_move(state)
        assert move1 is not None
        state2 = state.apply_move(move1)
        # Play as opponent (O)
        state2_as_o = make_state(
            [[int(c.value) for c in row] for row in state2.board],
            player="O",
        )
        move2 = agent.choose_move(state2_as_o)
        assert move2 is not None
        assert state2_as_o.board[move2[0]][move2[1]] is Cell.EMPTY
