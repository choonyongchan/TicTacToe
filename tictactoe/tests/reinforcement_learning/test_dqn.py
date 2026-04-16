"""Tests for DQNAgent."""
import numpy as np
import pytest

from tictactoe.agents.reinforcement_learning.dqn import DQNAgent
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player


def make_empty_state(n=3):
    board = Board.create(n)
    return GameState(board=board, current_player=Player.X, n=n, k=n)


def test_returns_legal_move_before_training():
    agent = DQNAgent(n=3)
    state = make_empty_state()
    move = agent.choose_move(state)
    r, c = move
    assert state.board[r][c] is Cell.EMPTY


def test_illegal_moves_are_masked():
    agent = DQNAgent(n=3)
    # Fill all but one cell
    board = Board.create(3)
    from tictactoe.core.types import Cell as C
    for r in range(3):
        for c in range(3):
            if not (r == 2 and c == 2):
                board[r][c] = C.X if (r + c) % 2 == 0 else C.O
    state = GameState(board=board, current_player=Player.X, n=3, k=3)
    move = agent.choose_move(state)
    assert move == (2, 2)


def test_nodes_visited_is_1():
    agent = DQNAgent(n=3)
    state = make_empty_state()
    agent.choose_move(state)
    assert state.nodes_visited == 1


def test_get_name_and_tier():
    agent = DQNAgent(n=3)
    assert "DQN" in agent.get_name()
    assert agent.get_tier() == 4


def test_save_load_roundtrip(tmp_path):
    import torch
    agent = DQNAgent(n=3)
    path = str(tmp_path / "dqn_model")
    agent.save(path)
    agent2 = DQNAgent(n=3)
    agent2.load(path)
    sd1 = agent._online.state_dict()
    sd2 = agent2._online.state_dict()
    for k in sd1:
        assert torch.allclose(sd1[k].float(), sd2[k].float()), f"Mismatch at key {k!r}"
