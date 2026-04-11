"""Tests for AlphaZeroAgent (previously tested via MCTSAlphaZeroLite)."""
from __future__ import annotations

import numpy as np
import pytest

from tictactoe.agents.reinforcement_learning.alphazero import AlphaZeroAgent
from tictactoe.benchmark.metrics import MatchConfig
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player, Result


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
    winner = Board.check_win_full(board, n, k)
    if winner is Player.X:
        result = Result.X_WINS
    elif winner is Player.O:
        result = Result.O_WINS
    elif Board.is_full(board):
        result = Result.DRAW
    else:
        result = Result.IN_PROGRESS
    return GameState(board=board, current_player=player, n=n, k=k,
                     last_move=last_move, result=result)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_legal_move_without_pretrained_network():
    agent = AlphaZeroAgent(n=3, num_simulations=10)
    state = make_state(["...", "...", "..."])
    move = agent.choose_move(state)
    r, c = move
    assert 0 <= r < 3 and 0 <= c < 3
    assert state.board[r][c] is Cell.EMPTY


def test_returns_legal_move_with_injected_network():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
    net = PolicyValueNetwork(3)
    agent = AlphaZeroAgent(n=3, net=net, num_simulations=10)
    state = make_state(["...", "...", "..."])
    move = agent.choose_move(state)
    r, c = move
    assert 0 <= r < 3 and 0 <= c < 3
    assert state.board[r][c] is Cell.EMPTY


def test_get_name_and_tier():
    agent = AlphaZeroAgent(n=3, num_simulations=5)
    name = agent.get_name()
    assert len(name) > 0
    assert agent.get_tier() == 4


def test_nodes_visited_positive():
    agent = AlphaZeroAgent(n=3, num_simulations=10)
    state = make_state(["...", "...", "..."])
    agent.choose_move(state)
    assert state.nodes_visited > 0


def test_picks_immediate_winning_move():
    # check_forced_move handles this before MCTS even runs
    agent = AlphaZeroAgent(n=3, num_simulations=20)
    state = make_state(["XX.", "...", "..."], player=Player.X, k=3)
    move = agent.choose_move(state)
    assert move == (0, 2)
