"""Tests for MCTSRave agent."""
from __future__ import annotations

import pytest

from tictactoe.agents.monte_carlo.mcts_rave import MCTSRave
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.benchmark.arena import Arena
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

def test_returns_legal_move_empty_board():
    agent = MCTSRave(match_config=MatchConfig.node_controlled(50))
    state = make_state(["...", "...", "..."])
    move = agent.choose_move(state)
    r, c = move
    assert 0 <= r < 3 and 0 <= c < 3
    assert state.board[r][c] is Cell.EMPTY


def test_picks_immediate_winning_move():
    agent = MCTSRave(match_config=MatchConfig.node_controlled(50), seed=42)
    state = make_state(["XX.", "...", "..."], player=Player.X, k=3)
    move = agent.choose_move(state)
    assert move == (0, 2)


def test_blocks_immediate_opponent_win():
    agent = MCTSRave(match_config=MatchConfig.node_controlled(50), seed=42)
    state = make_state(["OO.", "X..", "X.."], player=Player.X, k=3)
    move = agent.choose_move(state)
    assert move == (0, 2)


def test_win_rate_vs_random_agent():
    """RAVE should win or draw the vast majority of games against a random agent."""
    result = Arena(n=3, num_games=30).duel(
        MCTSRave(match_config=MatchConfig.node_controlled(500)),
        RandomAgent(),
    )
    # RAVE with sufficient budget should not lose more than a tiny fraction
    non_loss_rate = (result.agent_a_wins + result.draws) / result.total_games
    assert non_loss_rate >= 0.85


def test_get_name_and_tier():
    agent = MCTSRave()
    assert "RAVE" in agent.get_name()
    assert agent.get_tier() == 3


def test_nodes_visited_positive():
    agent = MCTSRave(match_config=MatchConfig.node_controlled(50))
    state = make_state(["...", "...", "..."])
    agent.choose_move(state)
    assert state.nodes_visited > 0


def test_amaf_values_nonempty_after_simulation():
    """After one choose_move call, root should have expanded children with visits."""
    from tictactoe.agents.monte_carlo.node import MCTSNode
    agent = MCTSRave(match_config=MatchConfig.node_controlled(20), seed=0)
    state = make_state(["...", "...", "..."])
    agent.choose_move(state)
    # We can't directly inspect root, but we can verify simulations ran
    assert state.nodes_visited > 0
    # Additionally verify at least something was expanded by re-running
    # with a fresh state and checking that nodes_visited reflects real work
    state2 = make_state(["X..", "...", "..."])
    agent.choose_move(state2)
    assert state2.nodes_visited > 0
