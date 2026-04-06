"""Tests for the Arena benchmarking engine."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.benchmark.arena import Arena
from tictactoe.benchmark.correctness import BruteForceOracle
from tictactoe.benchmark.metrics import MatchConfig, compute_elo
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import RANDOM_AGENT_WIN_THRESHOLD, Move, Player, Result
from tictactoe.benchmark.reporter import export_to_json, export_to_csv


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


class _NamedRandom(RandomAgent):
    """RandomAgent with a custom name, used only in tests."""

    def __init__(self, name: str, seed: int) -> None:
        super().__init__(seed=seed)
        self._custom_name = name

    def get_name(self) -> str:
        return self._custom_name


def _make_named_agents(count: int) -> list[BaseAgent]:
    """Create *count* RandomAgents each with a unique deterministic name.

    Args:
        count: Number of agents to create.

    Returns:
        A list of _NamedRandom instances with distinct names like "Random-0".
    """
    return [_NamedRandom(name=f"Random-{i}", seed=i) for i in range(count)]


# ---------------------------------------------------------------------------
# Perfect oracle for sanity-check tests
# ---------------------------------------------------------------------------


class PerfectAgent(BaseAgent):
    """Plays the first available winning move, then blocks, then random."""

    def choose_move(self, state: GameState) -> Move:
        board = state.board
        n = state.n
        k = state.k
        player = state.current_player

        move = (
            Board.get_winning_move(board, n, k, player)
            or Board.get_blocking_move(board, n, k, player)
            or Board.get_all_empty_cells(board)[0]
        )

        state.nodes_visited = 1
        state.max_depth_reached = 0
        state.prunings = 0
        state.compute_ebf()
        return move

    def get_name(self) -> str:
        return "PerfectAgent"

    def get_tier(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# duel
# ---------------------------------------------------------------------------


class TestDuel:
    """Tests for Arena.duel."""

    def test_correct_total_game_count(self) -> None:
        arena = Arena(n=3, num_games=10)
        result = arena.duel(RandomAgent(seed=1), RandomAgent(seed=2))
        assert result.total_games == 10

    def test_odd_num_games_raises(self) -> None:
        arena = Arena(n=3, num_games=7)
        with pytest.raises(ValueError):
            arena.duel(RandomAgent(), RandomAgent(), swap_sides=True)

    def test_swap_sides_does_not_double_games(self) -> None:
        arena = Arena(n=3, num_games=10)
        result = arena.duel(RandomAgent(seed=0), RandomAgent(seed=1), swap_sides=True)
        assert result.total_games == 10

    def test_wins_plus_draws_equals_total(self) -> None:
        arena = Arena(n=3, num_games=10)
        result = arena.duel(RandomAgent(seed=3), RandomAgent(seed=4))
        assert result.agent_a_wins + result.agent_b_wins + result.draws == 10


# ---------------------------------------------------------------------------
# round_robin
# ---------------------------------------------------------------------------


class TestRoundRobin:
    """Tests for Arena.round_robin."""

    def test_all_unique_pairs_covered(self) -> None:
        agents = _make_named_agents(4)
        arena = Arena(n=3, num_games=4)
        result = arena.round_robin(agents)

        expected_pairs = 6  # C(4,2)
        assert len(result.duel_matrix) == expected_pairs

    def test_no_self_duels(self) -> None:
        agents = _make_named_agents(3)
        arena = Arena(n=3, num_games=4)
        result = arena.round_robin(agents)
        for a_name, b_name in result.duel_matrix.keys():
            assert a_name != b_name

    def test_single_agent_raises(self) -> None:
        arena = Arena(n=3, num_games=4)
        with pytest.raises(ValueError):
            arena.round_robin([RandomAgent()])


# ---------------------------------------------------------------------------
# compute_elo
# ---------------------------------------------------------------------------


class TestComputeElo:
    """Tests for metrics.compute_elo."""

    def test_ratings_approximately_zero_sum(self) -> None:
        # With 2 agents starting at 1500, total should stay near 3000.
        # Elo is zero-sum per game, so across many games the sum stays at
        # 2 × initial_rating = 3000. Allow a generous tolerance because
        # 10 games is a small sample.
        arena = Arena(n=3, num_games=10)
        result = arena.duel(RandomAgent(seed=0), RandomAgent(seed=1))
        from tictactoe.benchmark.metrics import compute_elo
        ratings = compute_elo([result])
        total = sum(ratings.values())
        assert abs(total - 3000.0) < 1.0  # Elo is exactly zero-sum per game.


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Tests for Arena reproducibility given a fixed seed."""

    def test_same_seed_same_result(self) -> None:
        def run(seed: int) -> int:
            arena = Arena(n=3, num_games=10)
            arena.set_seed(seed)
            result = arena.duel(RandomAgent(seed=seed), RandomAgent(seed=seed + 1))
            return result.agent_a_wins

        assert run(42) == run(42)


# ---------------------------------------------------------------------------
# scalability_sweep
# ---------------------------------------------------------------------------


class TestScalabilitySweep:
    """Tests for Arena.scalability_sweep."""

    def test_returns_one_record_per_agent(self) -> None:
        agents = [RandomAgent(seed=0), RandomAgent(seed=1)]
        arena = Arena(n=3, num_games=4)
        records = arena.scalability_sweep(agents, board_sizes=[3, 5], games_per_size=4)
        assert len(records) == 2

    def test_record_has_data_per_size(self) -> None:
        arena = Arena(n=3, num_games=4)
        records = arena.scalability_sweep([RandomAgent(seed=0)], board_sizes=[3, 5], games_per_size=4)
        assert len(records[0].avg_depth_per_size) == 2
        assert len(records[0].avg_nodes_per_size) == 2


# ---------------------------------------------------------------------------
# sanity_check
# ---------------------------------------------------------------------------


class TestSanityCheck:
    """Tests for Arena.sanity_check."""

    def test_passes_for_near_perfect_agent(self) -> None:
        arena = Arena(n=3, num_games=50)
        arena.set_seed(42)
        result = arena.sanity_check(BruteForceOracle(), games=50)
        assert result["passed"] is True

    def test_fails_for_random_vs_random(self) -> None:
        # RandomAgent against RandomAgent on n=3 should win ~50% → fail.
        arena = Arena(n=3, num_games=50)
        arena.set_seed(7)
        result = arena.sanity_check(RandomAgent(seed=99), games=50)
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


class TestExport:
    """Tests for export_to_json and export_to_csv."""

    def test_export_to_json_valid(self) -> None:
        arena = Arena(n=3, num_games=4)
        result = arena.round_robin([RandomAgent(seed=0), RandomAgent(seed=1)])

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "result.json")
            export_to_json(result, path)
            with open(path) as f:
                data = json.load(f)
            assert "agents" in data

    def test_export_to_csv_header_and_rows(self) -> None:
        arena = Arena(n=3, num_games=4)
        result = arena.round_robin([RandomAgent(seed=0), RandomAgent(seed=1)])

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "result.csv")
            export_to_csv(result, path)
            content = Path(path).read_text()
            lines = content.strip().split("\n")
            # 1 header + 1 duel row for 2 agents.
            assert len(lines) >= 2
            assert "agent_a" in lines[0]
