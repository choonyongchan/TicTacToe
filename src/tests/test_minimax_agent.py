"""Tests for src/agents/minimax_agent.py."""
from __future__ import annotations

from src.agents.minimax_agent import MinimaxAgent
from src.core.board import Board
from src.core.state import State
from src.core.types import Player


def fresh_state() -> State:
    return State()


def state_with_moves(moves: list[tuple[int, int]]) -> State:
    s = State()
    for row, col in moves:
        s.apply(row, col)
    return s


class TestMinimaxAgentInit:
    def test_name(self):
        agent = MinimaxAgent(Player.X)
        assert agent.name == "MinimaxAgent"

    def test_maximizer_stored(self):
        agent = MinimaxAgent(Player.O)
        assert agent.maximizer == Player.O


class TestTerminalScore:
    """_terminal_score() returns +1/−1/0 relative to the maximizer."""

    def test_maximizer_wins_returns_1(self):
        # Row 0: X wins. Maximizer=X.
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
        agent = MinimaxAgent(Player.X)
        assert agent._terminal_score(state) == 1

    def test_opponent_wins_returns_neg1(self):
        # Col 0: O wins. Maximizer=X.
        state = state_with_moves([(0, 1), (0, 0), (0, 2), (1, 0), (2, 1), (2, 0)])
        agent = MinimaxAgent(Player.X)
        assert agent._terminal_score(state) == -1

    def test_draw_returns_0(self):
        # Full board, no winner
        state = state_with_moves([
            (0, 0), (0, 1),
            (0, 2), (1, 0),
            (1, 1), (2, 0),
            (1, 2), (2, 2),
            (2, 1),
        ])
        assert state.is_terminal()
        assert state.winner() is None
        agent = MinimaxAgent(Player.X)
        assert agent._terminal_score(state) == 0


# Shared fixture for the dummy tree position.
# Board after: X(0,0), O(0,1), X(1,1), O(2,2), X(1,0), O(2,0)
#   X | O | .
#   X | X | .
#   O | . | O
# It is X's turn. Maximizer = X. Empty cells: (0,2), (1,2), (2,1)
#
# Hand-traced minimax tree:
#   X(1,2) → row 1 = X,X,X → X WINS → score = 1
#   X(0,2) → O to move, empty: (1,2),(2,1)
#            O(2,1) → row 2 = O,O,O → O WINS → score = -1
#            O(1,2) → X to move, empty: (2,1)
#                     X(2,1) → board full, no win → DRAW → score = 0
#            minimize(-1, 0) = -1
#   X(2,1) → O to move, empty: (0,2),(1,2)
#            O(0,2) → X to move, empty: (1,2)
#                     X(1,2) → row 1 = X,X,X → X WINS → score = 1
#            O(1,2) → X to move, empty: (0,2)
#                     X(0,2) → board full, no win → DRAW → score = 0
#            minimize(1, 0) = 0
DUMMY_TREE_MOVES = [(0, 0), (0, 1), (1, 1), (2, 2), (1, 0), (2, 0)]


class TestMinimaxSmallTree:
    """Manually verified minimax values on a concrete 3-ply search tree."""

    def test_winning_branch_scores_1(self):
        # After X plays (1,2), row 1 is X,X,X → X wins immediately.
        state = state_with_moves(DUMMY_TREE_MOVES)
        state.apply(1, 2)
        agent = MinimaxAgent(Player.X)
        assert state.is_terminal(), "Expected terminal (X wins row 1)"
        assert agent._minimax(state) == 1

    def test_losing_branch_scores_neg1(self):
        # After X plays (0,2), optimal O play gives O the win.
        state = state_with_moves(DUMMY_TREE_MOVES)
        state.apply(0, 2)
        agent = MinimaxAgent(Player.X)
        assert agent._minimax(state) == -1

    def test_draw_branch_scores_0(self):
        # After X plays (2,1), optimal O play forces a draw.
        state = state_with_moves(DUMMY_TREE_MOVES)
        state.apply(2, 1)
        agent = MinimaxAgent(Player.X)
        assert agent._minimax(state) == 0

    def test_maximize_picks_best_child(self):
        # From the dummy position X's turn; _maximize() must return 1.
        state = state_with_moves(DUMMY_TREE_MOVES)
        agent = MinimaxAgent(Player.X)
        assert agent._maximize(state) == 1

    def test_minimize_picks_worst_for_maximizer(self):
        # After X(2,1): O sees scores [1,0] → minimize = 0.
        state = state_with_moves(DUMMY_TREE_MOVES)
        state.apply(2, 1)
        agent = MinimaxAgent(Player.X)
        assert agent._minimize(state) == 0


class TestActWinningMove:
    """act() must prefer an immediate win over any other move."""

    def test_picks_winning_move_on_dummy_tree(self):
        # From the dummy tree position, (1,2) is the only winning move.
        state = state_with_moves(DUMMY_TREE_MOVES)
        agent = MinimaxAgent(Player.X)
        assert agent.act(state) == (1, 2)

    def test_does_not_play_losing_move(self):
        state = state_with_moves(DUMMY_TREE_MOVES)
        agent = MinimaxAgent(Player.X)
        assert agent.act(state) != (0, 2)

    def test_wins_in_one_row(self):
        # Board: X,X,. / O,O,. / ...  X must play (0,2) to win.
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1)])
        agent = MinimaxAgent(Player.X)
        assert agent.act(state) == (0, 2)

    def test_wins_in_one_column(self):
        # Col 0: X at (0,0),(1,0). X plays (2,0) to win.
        state = state_with_moves([(0, 0), (0, 2), (1, 0), (1, 2)])
        agent = MinimaxAgent(Player.X)
        assert agent.act(state) == (2, 0)


class TestActBlockingMove:
    """act() must block the opponent's immediate win when no own win exists."""

    def test_blocks_opponent_row_win(self):
        # Board: O,O,. / . / X,X,.  X to move, no X win in 1, O wins with (0,2).
        state = state_with_moves([(2, 0), (0, 0), (2, 1), (0, 1)])
        agent = MinimaxAgent(Player.X)
        assert agent.act(state) == (0, 2)

    def test_blocks_opponent_diagonal_win(self):
        # O has main diagonal: (0,0),(1,1). O wins with (2,2). X must block.
        state = state_with_moves([(0, 2), (0, 0), (1, 2), (1, 1)])
        agent = MinimaxAgent(Player.X)
        assert agent.act(state) == (2, 2)


class TestActOptimalPlay:
    """MinimaxAgent plays perfectly — two MinimaxAgents must always draw."""

    def _play_game(self, agent_x: MinimaxAgent, agent_o: MinimaxAgent) -> Player | None:
        state = fresh_state()
        agents = {Player.X: agent_x, Player.O: agent_o}
        while not state.is_terminal():
            agent = agents[state.current_player]
            row, col = agent.act(state)
            state.apply(row, col)
        return state.winner()

    def test_two_minimax_agents_draw(self):
        agent_x = MinimaxAgent(Player.X)
        agent_o = MinimaxAgent(Player.O)
        result = self._play_game(agent_x, agent_o)
        assert result is None, f"Expected draw, got winner={result}"


class TestActValidMove:
    """act() always returns a legal move regardless of board state."""

    def test_returns_tuple(self):
        agent = MinimaxAgent(Player.X)
        move = agent.act(fresh_state())
        assert isinstance(move, tuple) and len(move) == 2

    def test_returns_in_bounds(self):
        agent = MinimaxAgent(Player.X)
        row, col = agent.act(fresh_state())
        assert 0 <= row <= 2 and 0 <= col <= 2

    def test_returns_empty_cell(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0)])
        agent = MinimaxAgent(Player.O)
        row, col = agent.act(state)
        assert Board.is_empty(state.board, row, col)

    def test_only_one_cell_left(self):
        # Fill 8 cells leaving only (2,2) empty, with no winner at any step.
        state = state_with_moves([
            (1, 0), (0, 0),
            (0, 1), (1, 1),
            (0, 2), (1, 2),
            (2, 0), (2, 1),
        ])
        assert not state.is_terminal()
        agent = MinimaxAgent(Player.X)
        assert agent.act(state) == (2, 2)
