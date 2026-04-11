"""Tests for tictactoe/benchmark/metrics.py.

Covers MatchConfig factories, aggregate_move_stats, compute_pruning_rate,
GameRecord, DuelResult, compute_elo, and RoundRobinResult.
"""
from __future__ import annotations

import json
import math

import pytest

from tictactoe.benchmark.metrics import (
    DuelResult,
    GameRecord,
    MatchConfig,
    MoveStats,
    RoundRobinResult,
    ScalabilityRecord,
    aggregate_move_stats,
    compute_elo,
    compute_pruning_rate,
)
from tictactoe.core.types import MatchMode, Player, Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _move_stats(
    nodes: int = 10,
    depth: int = 3,
    prunings: int = 2,
    time_ms: float = 5.0,
    ebf: float = 2.0,
    player: Player = Player.X,
    exceeded: bool = False,
) -> MoveStats:
    return MoveStats(
        move_number=1,
        player=player,
        move=(0, 0),
        nodes_visited=nodes,
        max_depth_reached=depth,
        prunings=prunings,
        pruning_rate=prunings / max(nodes, 1),
        time_taken_ms=time_ms,
        effective_branching_factor=ebf,
        time_limit_exceeded=exceeded,
    )


def _game_record(result: Result = Result.X_WINS) -> GameRecord:
    return GameRecord(
        agent_x_name="AgentA",
        agent_o_name="AgentB",
        n=3,
        k=3,
        match_config=MatchConfig.time_controlled(1000.0),
        result=result,
        total_moves=5,
        move_history=[(0, 0), (1, 1), (0, 1), (2, 2), (0, 2)],
        per_move_stats=[
            _move_stats(player=Player.X),
            _move_stats(player=Player.O),
            _move_stats(player=Player.X),
            _move_stats(player=Player.O),
            _move_stats(player=Player.X),
        ],
    )


def _duel_result(a_wins: int = 7, b_wins: int = 2, draws: int = 1) -> DuelResult:
    return DuelResult(
        agent_a_name="A",
        agent_b_name="B",
        agent_a_wins=a_wins,
        agent_b_wins=b_wins,
        draws=draws,
        total_games=a_wins + b_wins + draws,
        n=3,
        k=3,
        match_config=MatchConfig.time_controlled(),
        games=[_game_record()],
    )


# ---------------------------------------------------------------------------
# MatchConfig factories
# ---------------------------------------------------------------------------


class TestMatchConfigFactories:
    """Tests for MatchConfig factory class methods."""

    def test_time_controlled_factory(self) -> None:
        cfg = MatchConfig.time_controlled(500.0)
        assert cfg.mode is MatchMode.TIME_CONTROLLED
        assert cfg.time_limit_ms == 500.0

    def test_node_controlled_factory(self) -> None:
        cfg = MatchConfig.node_controlled(200)
        assert cfg.mode is MatchMode.NODE_CONTROLLED
        assert cfg.node_budget == 200

    def test_depth_controlled_factory(self) -> None:
        cfg = MatchConfig.depth_controlled(6)
        assert cfg.mode is MatchMode.DEPTH_CONTROLLED
        assert cfg.fixed_depth == 6

    def test_time_controlled_default_value(self) -> None:
        cfg = MatchConfig.time_controlled()
        assert cfg.time_limit_ms > 0


# ---------------------------------------------------------------------------
# aggregate_move_stats
# ---------------------------------------------------------------------------


class TestAggregateMoveStats:
    """Tests for the aggregate_move_stats() free function."""

    EXPECTED_KEYS = {
        "mean_nodes", "std_nodes",
        "mean_time_ms", "std_time_ms",
        "mean_prunings", "std_prunings",
        "mean_ebf", "std_ebf",
        "mean_depth", "std_depth",
    }

    def test_empty_list_returns_all_zeros(self) -> None:
        result = aggregate_move_stats([])
        for key in self.EXPECTED_KEYS:
            assert result[key] == 0.0, f"{key} should be 0.0 for empty input"

    def test_all_expected_keys_present(self) -> None:
        result = aggregate_move_stats([_move_stats()])
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_single_item_std_is_zero(self) -> None:
        result = aggregate_move_stats([_move_stats(nodes=20)])
        assert result["std_nodes"] == 0.0
        assert result["std_time_ms"] == 0.0

    def test_single_item_mean_equals_value(self) -> None:
        result = aggregate_move_stats([_move_stats(nodes=42, time_ms=7.5)])
        assert result["mean_nodes"] == pytest.approx(42.0)
        assert result["mean_time_ms"] == pytest.approx(7.5)

    def test_two_items_mean_correct(self) -> None:
        stats = [_move_stats(nodes=10), _move_stats(nodes=20)]
        result = aggregate_move_stats(stats)
        assert result["mean_nodes"] == pytest.approx(15.0)

    def test_std_dev_nonzero_for_different_values(self) -> None:
        stats = [_move_stats(nodes=10), _move_stats(nodes=20)]
        result = aggregate_move_stats(stats)
        assert result["std_nodes"] > 0.0


# ---------------------------------------------------------------------------
# compute_pruning_rate
# ---------------------------------------------------------------------------


class TestComputePruningRate:
    """Tests for the compute_pruning_rate() free function."""

    def test_empty_list_returns_zero(self) -> None:
        assert compute_pruning_rate([]) == 0.0

    def test_all_zero_nodes_excluded(self) -> None:
        stats = [_move_stats(nodes=0, prunings=5)]
        assert compute_pruning_rate(stats) == 0.0

    def test_correct_rate_single_entry(self) -> None:
        stats = [_move_stats(nodes=10, prunings=4)]
        assert compute_pruning_rate(stats) == pytest.approx(0.4)

    def test_average_over_multiple_entries(self) -> None:
        # rates: 0.5 and 0.25 → average 0.375
        stats = [
            _move_stats(nodes=10, prunings=5),
            _move_stats(nodes=20, prunings=5),
        ]
        assert compute_pruning_rate(stats) == pytest.approx(0.375)

    def test_zero_node_entries_are_excluded(self) -> None:
        # Only the second entry (nodes=10, prunings=5) counts → rate = 0.5
        stats = [
            _move_stats(nodes=0, prunings=99),
            _move_stats(nodes=10, prunings=5),
        ]
        assert compute_pruning_rate(stats) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# GameRecord
# ---------------------------------------------------------------------------


class TestGameRecord:
    """Tests for GameRecord methods."""

    def test_get_stats_for_player_x(self) -> None:
        record = _game_record()
        x_stats = record.get_stats_for_player("AgentA")
        assert all(s.player is Player.X for s in x_stats)
        assert len(x_stats) == 3  # X moves at positions 0, 2, 4

    def test_get_stats_for_player_o(self) -> None:
        record = _game_record()
        o_stats = record.get_stats_for_player("AgentB")
        assert all(s.player is Player.O for s in o_stats)
        assert len(o_stats) == 2  # O moves at positions 1, 3

    def test_to_dict_json_serializable(self) -> None:
        record = _game_record()
        as_dict = record.to_dict()
        json_str = json.dumps(as_dict)  # must not raise
        assert isinstance(json_str, str)

    def test_to_dict_has_all_keys(self) -> None:
        record = _game_record()
        as_dict = record.to_dict()
        for key in ["agent_x_name", "agent_o_name", "n", "k", "match_mode",
                    "result", "total_moves", "move_history", "per_move_stats"]:
            assert key in as_dict, f"Missing key: {key}"

    def test_to_dict_result_is_string(self) -> None:
        record = _game_record(result=Result.DRAW)
        assert record.to_dict()["result"] == "DRAW"


# ---------------------------------------------------------------------------
# DuelResult
# ---------------------------------------------------------------------------


class TestDuelResult:
    """Tests for DuelResult.win_rate_a/b and to_dict."""

    def test_win_rate_a_zero_games(self) -> None:
        dr = DuelResult(
            agent_a_name="A", agent_b_name="B",
            agent_a_wins=0, agent_b_wins=0, draws=0, total_games=0,
            n=3, k=3, match_config=MatchConfig.time_controlled(),
        )
        assert dr.win_rate_a() == 0.0
        assert dr.win_rate_b() == 0.0

    def test_win_rate_a_correct_fraction(self) -> None:
        dr = _duel_result(a_wins=7, b_wins=2, draws=1)
        assert dr.win_rate_a() == pytest.approx(7 / 10)

    def test_win_rate_b_correct_fraction(self) -> None:
        dr = _duel_result(a_wins=7, b_wins=2, draws=1)
        assert dr.win_rate_b() == pytest.approx(2 / 10)

    def test_to_dict_json_serializable(self) -> None:
        dr = _duel_result()
        as_dict = dr.to_dict()
        json_str = json.dumps(as_dict)
        assert isinstance(json_str, str)

    def test_to_dict_win_rates_correct(self) -> None:
        dr = _duel_result(a_wins=6, b_wins=4, draws=0)
        d = dr.to_dict()
        assert d["win_rate_a"] == pytest.approx(0.6)
        assert d["win_rate_b"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# compute_elo
# ---------------------------------------------------------------------------


class TestComputeElo:
    """Tests for the compute_elo() free function."""

    def test_empty_results_returns_empty(self) -> None:
        ratings = compute_elo([])
        assert ratings == {}

    def test_new_agents_start_at_initial_rating(self) -> None:
        # With one perfectly drawn game, both ratings shift by equal amounts.
        dr = _duel_result(a_wins=0, b_wins=0, draws=1)
        dr.games = [_game_record(result=Result.DRAW)]
        ratings = compute_elo([dr], initial_rating=1500.0)
        # Both start at 1500; a draw between equals → no change
        assert abs(ratings.get("A", 1500.0) - 1500.0) < 1.0
        assert abs(ratings.get("B", 1500.0) - 1500.0) < 1.0

    def test_ratings_zero_sum(self) -> None:
        """Sum of two agents' ratings must equal 2 × initial_rating."""
        dr = _duel_result(a_wins=3, b_wins=7, draws=0)
        # Provide real game records for compute_elo to iterate over.
        games = [_game_record(Result.X_WINS)] * 3 + [_game_record(Result.O_WINS)] * 7
        dr.games = games
        ratings = compute_elo([dr], initial_rating=1500.0)
        total = sum(ratings.values())
        assert abs(total - 3000.0) < 0.5  # Elo is exactly zero-sum per game


# ---------------------------------------------------------------------------
# ScalabilityRecord fields
# ---------------------------------------------------------------------------


class TestScalabilityRecord:
    """Tests for ScalabilityRecord construction."""

    def test_default_lists_are_empty(self) -> None:
        rec = ScalabilityRecord(
            agent_name="Test",
            board_sizes=[3, 5],
            match_config=MatchConfig.time_controlled(),
        )
        assert rec.avg_nodes_per_size == []
        assert rec.avg_ebf_per_size == []
        assert rec.budget_exhausted_per_size == []
        assert rec.total_moves_per_size == []


# ---------------------------------------------------------------------------
# RoundRobinResult
# ---------------------------------------------------------------------------


class TestRoundRobinResult:
    """Tests for RoundRobinResult.get_win_rate_matrix()."""

    def test_win_rate_matrix_keys_match_duel_matrix(self) -> None:
        rr = RoundRobinResult(
            agents=["A", "B", "C"],
            n=3,
            match_config=MatchConfig.time_controlled(),
            duel_matrix={
                ("A", "B"): _duel_result(a_wins=6, b_wins=4, draws=0),
                ("A", "C"): _duel_result(a_wins=5, b_wins=5, draws=0),
                ("B", "C"): _duel_result(a_wins=8, b_wins=2, draws=0),
            },
        )
        matrix = rr.get_win_rate_matrix()
        assert set(matrix.keys()) == {("A", "B"), ("A", "C"), ("B", "C")}

    def test_win_rate_matrix_values_correct(self) -> None:
        rr = RoundRobinResult(
            agents=["A", "B"],
            n=3,
            match_config=MatchConfig.time_controlled(),
            duel_matrix={("A", "B"): _duel_result(a_wins=7, b_wins=3, draws=0)},
        )
        matrix = rr.get_win_rate_matrix()
        assert matrix[("A", "B")] == pytest.approx(0.7)
