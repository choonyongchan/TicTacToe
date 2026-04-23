"""Tests for src/agents/minimax_alphabeta_agent.py."""
from __future__ import annotations

from src.agents.minimax_alphabeta_agent import MinimaxAlphaBetaAgent
from src.core.types import Player
from src.tests.test_helper import fresh_state, state_with_moves, PUZZLE_3X3, PUZZLE_4X4, PUZZLE_5X5


class TestMinimaxAlphaBetaAgentInit:
    def test_name(self):
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent.name == "MinimaxAlphaBetaAgent"

    def test_maximizer_stored(self):
        agent = MinimaxAlphaBetaAgent(Player.O)
        assert agent.maximizer == Player.O


class TestTerminalScore:
    """_terminal_score() returns +1/−1/0 relative to the maximizer."""

    def test_maximizer_wins_returns_1(self):
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._terminal_score(state) == 1

    def test_opponent_wins_returns_neg1(self):
        state = state_with_moves([(0, 1), (0, 0), (0, 2), (1, 0), (2, 1), (2, 0)])
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._terminal_score(state) == -1

    def test_draw_returns_0(self):
        state = state_with_moves([
            (0, 0), (0, 1),
            (0, 2), (1, 0),
            (1, 1), (2, 0),
            (1, 2), (2, 2),
            (2, 1),
        ])
        assert state.is_terminal()
        assert state.winner() is None
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._terminal_score(state) == 0


DUMMY_TREE_MOVES = PUZZLE_3X3.moves


class TestMinimaxAlphaBetaSmallTree:
    """Manually verified alpha-beta values on a concrete 3-ply search tree."""

    def test_winning_branch_scores_1(self):
        state = state_with_moves(DUMMY_TREE_MOVES)
        state.apply(1, 2)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert state.is_terminal(), "Expected terminal (X wins row 1)"
        assert agent._minimax(state, -2, 2) == 1

    def test_losing_branch_scores_neg1(self):
        state = state_with_moves(DUMMY_TREE_MOVES)
        state.apply(0, 2)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._minimax(state, -2, 2) == -1

    def test_draw_branch_scores_0(self):
        state = state_with_moves(DUMMY_TREE_MOVES)
        state.apply(2, 1)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._minimax(state, -2, 2) == 0

    def test_maximize_picks_best_child(self):
        state = state_with_moves(DUMMY_TREE_MOVES)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._maximize(state, -2, 2) == 1

    def test_minimize_picks_worst_for_maximizer(self):
        # After X(2,1): O sees scores [1, 0] → minimize = 0
        state = state_with_moves(DUMMY_TREE_MOVES)
        state.apply(2, 1)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._minimize(state, -2, 2) == 0


class TestAlphaBetaPruning:
    """Verify alpha-beta window bounds produce correct values and trigger pruning."""

    # --- _maximize tests (X to move from DUMMY root) ---

    def test_maximize_full_window_returns_1(self):
        # No pruning with full [-2, 2] window; best child is X(1,2)=1.
        state = state_with_moves(DUMMY_TREE_MOVES)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._maximize(state, -2, 2) == 1

    def test_maximize_beta_cutoff_after_win(self):
        # beta=0: after X(1,2) scores 1 >= beta=0, prune X(2,1). Still returns 1.
        state = state_with_moves(DUMMY_TREE_MOVES)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._maximize(state, -2, 0) == 1

    def test_maximize_beta_cutoff_at_first_child(self):
        # beta=-1: X(0,2) scores -1 >= beta=-1, prune immediately. Returns -1.
        state = state_with_moves(DUMMY_TREE_MOVES)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._maximize(state, -2, -1) == -1

    # --- _minimize tests (O to move after X plays (2,1)) ---
    # O's children: O(0,2)→X wins→1, O(1,2)→draw→0

    def test_minimize_full_window_returns_0(self):
        # No pruning with full window; O picks min(1, 0) = 0.
        state = state_with_moves(DUMMY_TREE_MOVES)
        state.apply(2, 1)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._minimize(state, -2, 2) == 0

    def test_minimize_alpha_cutoff_at_first_child(self):
        # alpha=1: O(0,2) gives score=1; 1 <= alpha=1, prune O(1,2). Returns 1.
        state = state_with_moves(DUMMY_TREE_MOVES)
        state.apply(2, 1)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._minimize(state, 1, 2) == 1

    def test_minimize_alpha_cutoff_at_second_child(self):
        # alpha=0: O(0,2)=1 ok (no prune), O(1,2)=0; 0 <= alpha=0, prune. Returns 0.
        state = state_with_moves(DUMMY_TREE_MOVES)
        state.apply(2, 1)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._minimize(state, 0, 2) == 0

    # --- consistency: alpha-beta must match pure minimax values ---

    def test_minimax_terminal_win_consistent(self):
        state = state_with_moves(DUMMY_TREE_MOVES)
        state.apply(1, 2)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._minimax(state, -2, 2) == 1

    def test_minimax_losing_branch_consistent(self):
        state = state_with_moves(DUMMY_TREE_MOVES)
        state.apply(0, 2)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._minimax(state, -2, 2) == -1

    def test_minimax_draw_branch_consistent(self):
        state = state_with_moves(DUMMY_TREE_MOVES)
        state.apply(2, 1)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent._minimax(state, -2, 2) == 0


class TestActWinningMove:
    """act() must prefer an immediate win over any other move."""

    def test_picks_winning_move_on_dummy_tree(self):
        state = state_with_moves(DUMMY_TREE_MOVES)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent.act(state) == (1, 2)

    def test_does_not_play_losing_move(self):
        state = state_with_moves(DUMMY_TREE_MOVES)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent.act(state) != (0, 2)

    def test_wins_in_one_row(self):
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1)])
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent.act(state) == (0, 2)

    def test_wins_in_one_column(self):
        state = state_with_moves([(0, 0), (0, 2), (1, 0), (1, 2)])
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent.act(state) == (2, 0)


class TestActBlockingMove:
    """act() must block the opponent's immediate win when no own win exists."""

    def test_blocks_opponent_row_win(self):
        state = state_with_moves([(2, 0), (0, 0), (2, 1), (0, 1)])
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent.act(state) == (0, 2)

    def test_blocks_opponent_diagonal_win(self):
        state = state_with_moves([(0, 2), (0, 0), (1, 2), (1, 1)])
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent.act(state) == (2, 2)


class TestActOptimalPlay:
    """MinimaxAlphaBetaAgent plays perfectly — two agents must always draw."""

    def _play_game(
        self,
        agent_x: MinimaxAlphaBetaAgent,
        agent_o: MinimaxAlphaBetaAgent,
    ) -> Player | None:
        state = fresh_state()
        agents = {Player.X: agent_x, Player.O: agent_o}
        while not state.is_terminal():
            agent = agents[state.current_player]
            row, col = agent.act(state)
            state.apply(row, col)
        return state.winner()

    def test_two_agents_draw(self):
        agent_x = MinimaxAlphaBetaAgent(Player.X)
        agent_o = MinimaxAlphaBetaAgent(Player.O)
        result = self._play_game(agent_x, agent_o)
        assert result is None, f"Expected draw, got winner={result}"


class TestActValidMove:
    """act() always returns a legal move regardless of board state."""

    def test_returns_tuple(self):
        agent = MinimaxAlphaBetaAgent(Player.X)
        move = agent.act(fresh_state())
        assert isinstance(move, tuple) and len(move) == 2

    def test_returns_in_bounds(self):
        agent = MinimaxAlphaBetaAgent(Player.X)
        row, col = agent.act(fresh_state())
        assert 0 <= row <= 2 and 0 <= col <= 2

    def test_returns_empty_cell(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0)])
        agent = MinimaxAlphaBetaAgent(Player.O)
        row, col = agent.act(state)
        assert state.board.is_empty(row, col)

    def test_only_one_cell_left(self):
        state = state_with_moves([
            (1, 0), (0, 0),
            (0, 1), (1, 1),
            (0, 2), (1, 2),
            (2, 0), (2, 1),
        ])
        assert not state.is_terminal()
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent.act(state) == (2, 2)


class TestActLargerBoards:
    """act() picks the correct move on 4×4 and 5×5 puzzle positions."""

    def test_4x4_picks_best_move(self):
        state = state_with_moves(PUZZLE_4X4.moves, PUZZLE_4X4.n, PUZZLE_4X4.k)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent.act(state) == PUZZLE_4X4.best_move

    def test_5x5_picks_best_move(self):
        state = state_with_moves(PUZZLE_5X5.moves, PUZZLE_5X5.n, PUZZLE_5X5.k)
        agent = MinimaxAlphaBetaAgent(Player.X)
        assert agent.act(state) == PUZZLE_5X5.best_move
