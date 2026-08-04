"""Tests for tictactoe/cli.py's game-runner orchestration."""

from __future__ import annotations

import time

import pytest

from tictactoe.cli import AGENTS, Main, _TT_AGENTS
from tictactoe.agents.base_agent import BaseAgent
from tictactoe.core.state import State
from tictactoe.core.tt_state import TTState


class TestAgentRegistry:
    def test_tt_agents_is_subset_of_agents(self):
        assert _TT_AGENTS <= AGENTS.keys()

    @pytest.mark.parametrize("key", list(AGENTS))
    def test_factory_builds_a_base_agent(self, key):
        agent = AGENTS[key]("X", 9)
        assert isinstance(agent, BaseAgent)


class TestInit:
    def test_uses_tt_state_for_tt_agents(self):
        game = Main(n=3, k=3, agent="mtdf", verbose=False)
        assert isinstance(game._state, TTState)

    def test_uses_plain_state_for_other_agents(self):
        game = Main(n=3, k=3, agent="random", verbose=False)
        assert isinstance(game._state, State)
        assert not isinstance(game._state, TTState)


class TestGameLoop:
    def test_plays_to_a_terminal_result(self):
        game = Main(n=3, k=3, agent="random", verbose=False)
        winner_label = game._game_loop(deadline=time.time() + 30)
        assert winner_label in ("X", "O", "Draw")
        assert game._state.is_terminal()

    def test_stops_early_when_deadline_already_passed(self):
        game = Main(n=3, k=3, agent="random", verbose=False)
        winner_label = game._game_loop(deadline=time.time() - 1)
        assert winner_label == "Timeout"
        assert not game._state.is_terminal()


class TestRun:
    def test_run_completes_and_prints_result(self, capsys):
        game = Main(n=3, k=3, agent="random", verbose=False)
        game.run()
        out = capsys.readouterr().out
        assert "Winner:" in out
        assert "Algorithm: RandomAgent" in out

    def test_run_reports_timeout_when_budget_exhausted(self, capsys):
        game = Main(n=3, k=3, agent="random", verbose=False)
        game._TIMEOUT = -1  # already-expired budget
        game.run()
        assert "Winner: Timeout" in capsys.readouterr().out
