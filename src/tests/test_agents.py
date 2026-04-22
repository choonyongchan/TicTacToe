"""Tests for src/agents: RandomAgent."""
from __future__ import annotations

from unittest.mock import patch

from src.agents.random_agent import RandomAgent
from src.core.board import Board as board_mod
from src.core.state import State
from src.core.types import Player


def fresh_state() -> State:
    return State()


def state_with_moves(moves: list[tuple[int, int]]) -> State:
    s = State()
    for row, col in moves:
        s.apply(row, col)
    return s


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------

class TestRandomAgentInit:
    def test_name(self):
        assert RandomAgent().name == "RandomAgent"


class TestRandomAgentAct:
    def test_act_returns_tuple(self):
        agent = RandomAgent()
        move = agent.act(fresh_state())
        assert isinstance(move, tuple) and len(move) == 2

    def test_act_returns_valid_row(self):
        agent = RandomAgent()
        row, _ = agent.act(fresh_state())
        assert 0 <= row <= 2

    def test_act_returns_valid_col(self):
        agent = RandomAgent()
        _, col = agent.act(fresh_state())
        assert 0 <= col <= 2

    def test_act_returns_empty_cell(self):
        agent = RandomAgent()
        state = fresh_state()
        row, col = agent.act(state)
        assert board_mod.is_empty(state.board, row, col)

    def test_act_never_returns_occupied_cell(self):
        # Fill all cells except (2, 2); act() must return (2, 2).
        moves = [
            (0, 0), (0, 1),
            (0, 2), (1, 0),
            (1, 1), (1, 2),
            (2, 0), (2, 1),
        ]
        state = state_with_moves(moves)
        agent = RandomAgent()
        assert agent.act(state) == (2, 2)

    def test_act_with_one_empty_cell(self):
        moves = [
            (0, 0), (0, 1),
            (0, 2), (1, 0),
            (1, 1), (1, 2),
            (2, 0), (2, 1),
        ]
        state = state_with_moves(moves)
        agent = RandomAgent()
        for _ in range(20):
            assert agent.act(state) == (2, 2)

    def test_act_on_fresh_board_covers_all_cells(self):
        agent = RandomAgent()
        seen: set[tuple[int, int]] = set()
        for _ in range(500):
            seen.add(agent.act(fresh_state()))
        assert len(seen) == 9


class TestRandomAgentActMocked:
    def test_act_calls_random_choice_with_empty_cells(self):
        state = fresh_state()
        expected_empty = board_mod.get_empty_cells(state.board)
        with patch("src.agents.random_agent.random.choice", return_value=(0, 0)) as mock_choice:
            RandomAgent().act(state)
            mock_choice.assert_called_once_with(expected_empty)
