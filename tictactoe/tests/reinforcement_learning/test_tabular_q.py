"""Tests for TabularQAgent."""
import pytest
from tictactoe.agents.reinforcement_learning.tabular_q import TabularQAgent
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player, Result


def make_empty_state(n=3, k=3):
    board = Board.create(n)
    return GameState(board=board, current_player=Player.X, n=n, k=k)


def test_raises_for_n_greater_than_3():
    with pytest.raises(NotImplementedError):
        TabularQAgent(n=5)


def test_raises_in_choose_move_for_n_greater_than_3():
    # Create agent with n=3 then manually set n=5 to test choose_move guard
    agent = TabularQAgent(n=3)
    agent.n = 5
    state = make_empty_state(n=5, k=5)
    with pytest.raises(NotImplementedError):
        agent.choose_move(state)


def test_q_table_empty_before_training():
    agent = TabularQAgent(n=3)
    assert len(agent._q_table) == 0


def test_choose_move_returns_legal_move():
    agent = TabularQAgent(n=3)
    state = make_empty_state()
    move = agent.choose_move(state)
    r, c = move
    assert 0 <= r < 3 and 0 <= c < 3
    assert state.board[r][c] is Cell.EMPTY


def test_nodes_visited_is_1():
    agent = TabularQAgent(n=3)
    state = make_empty_state()
    agent.choose_move(state)
    assert state.nodes_visited == 1


def test_choose_move_fast():
    import time
    agent = TabularQAgent(n=3)
    state = make_empty_state()
    start = time.perf_counter()
    agent.choose_move(state)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1  # Must be faster than 100ms


def test_get_name_and_tier():
    agent = TabularQAgent(n=3)
    assert "TabularQ" in agent.get_name()
    assert agent.get_tier() == 4


def test_save_load_roundtrip(tmp_path):
    agent = TabularQAgent(n=3)
    # Manually add a Q-value to verify persistence
    agent._q_table[42] = {0: 0.75}
    path = str(tmp_path / "q_table.pkl")
    agent.save(path)
    agent2 = TabularQAgent(n=3)
    agent2.load(path)
    assert agent2._q_table.get(42, {}).get(0) == pytest.approx(0.75)
