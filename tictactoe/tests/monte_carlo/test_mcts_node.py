"""Tests for MCTSNode."""
from __future__ import annotations

import math
import pytest

from tictactoe.agents.monte_carlo.node import MCTSNode
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, INF, Player, Result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_state(board_rows, player=Player.X, k=None):
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
    # Use full-board scan so terminal is detected regardless of which cell
    # is iterated last (Board.is_terminal only checks lines through last_move).
    winner = Board.check_win_full(board, n, k)
    if winner is Player.X:
        result = Result.X_WINS
    elif winner is Player.O:
        result = Result.O_WINS
    elif Board.is_full(board):
        result = Result.DRAW
    else:
        result = Result.IN_PROGRESS
    return GameState(
        board=board, current_player=player, n=n, k=k,
        last_move=last_move, result=result
    )


# ---------------------------------------------------------------------------
# ucb1_score
# ---------------------------------------------------------------------------

def test_ucb1_score_unvisited_is_inf():
    state = make_state(["...", "...", "..."])
    node = MCTSNode(state)
    assert node.ucb1_score() == INF


def test_ucb1_score_correct_for_visited():
    state = make_state(["...", "...", "..."])
    parent = MCTSNode(state)
    child = MCTSNode(state, parent=parent)
    parent.visits = 10
    child.visits = 4
    child.value = 3.0
    # Q = 3/4 = 0.75
    # exploration = 1.414 * sqrt(ln(10)/4) = 1.414 * sqrt(2.302585/4)
    expected_q = 3.0 / 4
    expected_exp = 1.414 * math.sqrt(math.log(10) / 4)
    expected = expected_q + expected_exp
    assert abs(child.ucb1_score(1.414) - expected) < 1e-9


# ---------------------------------------------------------------------------
# best_child / most_visited_child
# ---------------------------------------------------------------------------

def test_best_child_returns_highest_ucb1():
    state = make_state(["...", "X..", "..."])
    parent = MCTSNode(state)
    parent.visits = 20

    child_a = MCTSNode(state, parent=parent, move=(0, 0))
    child_a.visits = 5
    child_a.value = 2.0

    child_b = MCTSNode(state, parent=parent, move=(0, 1))
    child_b.visits = 2
    child_b.value = 1.0

    parent.children = [child_a, child_b]
    parent.untried_moves.clear()

    best = parent.best_child(1.414)
    # UCB1 for child_b should be higher (fewer visits, higher upper bound)
    assert best is child_b or child_a.ucb1_score() >= child_b.ucb1_score()
    # Just ensure best_child actually returns the max
    assert best.ucb1_score() == max(child_a.ucb1_score(), child_b.ucb1_score())


def test_most_visited_child_returns_highest_visits():
    state = make_state(["...", "X..", "..."])
    parent = MCTSNode(state)

    child_a = MCTSNode(state, parent=parent, move=(0, 0))
    child_a.visits = 10

    child_b = MCTSNode(state, parent=parent, move=(0, 1))
    child_b.visits = 3

    parent.children = [child_a, child_b]
    parent.untried_moves.clear()

    assert parent.most_visited_child() is child_a


# ---------------------------------------------------------------------------
# expand
# ---------------------------------------------------------------------------

def test_expand_reduces_untried_moves():
    state = make_state(["...", "X..", "..."])
    node = MCTSNode(state)
    initial_count = len(node.untried_moves)
    assert initial_count > 0
    child = node.expand()
    assert len(node.untried_moves) == initial_count - 1
    assert child in node.children


def test_expand_returns_new_child():
    state = make_state(["...", "X..", "..."])
    node = MCTSNode(state)
    child = node.expand()
    assert isinstance(child, MCTSNode)
    assert child.parent is node
    assert child.move is not None


def test_expand_on_fully_expanded_raises():
    state = make_state(["...", "X..", "..."])
    node = MCTSNode(state)
    # Exhaust all untried moves
    while node.untried_moves:
        node.expand()
    with pytest.raises(ValueError):
        node.expand()


# ---------------------------------------------------------------------------
# backpropagate
# ---------------------------------------------------------------------------

def test_backpropagate_increments_visits_at_ancestors():
    state = make_state(["...", "...", "..."])
    root = MCTSNode(state)
    child = MCTSNode(state, parent=root)
    grandchild = MCTSNode(state, parent=child)

    grandchild.backpropagate(1.0)

    assert grandchild.visits == 1
    assert child.visits == 1
    assert root.visits == 1


def test_backpropagate_flips_sign_zero_sum():
    """Result of +1 at grandchild should become -1 at child (parent)."""
    state = make_state(["...", "...", "..."])
    root = MCTSNode(state)
    child = MCTSNode(state, parent=root)
    grandchild = MCTSNode(state, parent=child)

    grandchild.backpropagate(1.0)

    # grandchild got +1
    assert grandchild.value == pytest.approx(1.0)
    # child got -1 (flip)
    assert child.value == pytest.approx(-1.0)
    # root got +1 (flip again)
    assert root.value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# is_terminal
# ---------------------------------------------------------------------------

def test_is_terminal_returns_true_for_terminal_state():
    # X wins on 3x3
    state = make_state(["XXX", "OO.", "..."])
    node = MCTSNode(state)
    assert node.is_terminal()


def test_is_terminal_returns_false_for_ongoing():
    state = make_state(["...", "...", "..."])
    node = MCTSNode(state)
    assert not node.is_terminal()


# ---------------------------------------------------------------------------
# RAVE fields
# ---------------------------------------------------------------------------

def test_rave_fields_init_to_zero():
    state = make_state(["...", "...", "..."])
    node = MCTSNode(state)
    assert node.rave_visits == 0
    assert node.rave_value == pytest.approx(0.0)
    assert node.amaf_values == {}
    assert node.amaf_visits == {}


def test_rave_score_zero_when_no_visits():
    state = make_state(["...", "...", "..."])
    node = MCTSNode(state)
    assert node.rave_score((0, 0)) == pytest.approx(0.0)
