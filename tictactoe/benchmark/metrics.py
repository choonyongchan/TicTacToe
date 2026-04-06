"""Benchmark data structures and statistical utilities.

Defines all dataclasses used to record, aggregate, and compare benchmark
results. Also provides free functions for Elo computation, move-stat
aggregation, and pruning-rate computation.

This module has no dependency on agent or evaluation code — it deals only
with numerical results and the MatchMode enum from core.types.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from tictactoe.core.types import (
    DEFAULT_NODE_BUDGET,
    DEFAULT_TIME_LIMIT_MS,
    MatchMode,
    Move,
    Player,
    Result,
)


# ---------------------------------------------------------------------------
# Match configuration
# ---------------------------------------------------------------------------


@dataclass
class MatchConfig:
    """Controls how move budgets are enforced during a match.

    Attributes:
        mode: Which budget type to enforce (time, nodes, or fixed depth).
        time_limit_ms: Per-move wall-clock budget in milliseconds.
            Only used when mode is TIME_CONTROLLED.
        node_budget: Per-move node-visit limit.
            Only used when mode is NODE_CONTROLLED.
        fixed_depth: Search depth limit per move.
            Only used when mode is DEPTH_CONTROLLED.
    """

    mode: MatchMode
    time_limit_ms: float = DEFAULT_TIME_LIMIT_MS
    node_budget: int = DEFAULT_NODE_BUDGET
    fixed_depth: int = 4

    # --- Convenience factory methods ---

    @classmethod
    def time_controlled(cls, ms: float = DEFAULT_TIME_LIMIT_MS) -> MatchConfig:
        """Create a TIME_CONTROLLED configuration.

        Args:
            ms: Per-move time limit in milliseconds.

        Returns:
            A MatchConfig with mode=TIME_CONTROLLED.
        """
        return cls(mode=MatchMode.TIME_CONTROLLED, time_limit_ms=ms)

    @classmethod
    def node_controlled(cls, budget: int = DEFAULT_NODE_BUDGET) -> MatchConfig:
        """Create a NODE_CONTROLLED configuration.

        Args:
            budget: Per-move node-visit limit.

        Returns:
            A MatchConfig with mode=NODE_CONTROLLED.
        """
        return cls(mode=MatchMode.NODE_CONTROLLED, node_budget=budget)

    @classmethod
    def depth_controlled(cls, depth: int = 4) -> MatchConfig:
        """Create a DEPTH_CONTROLLED configuration for ablation studies.

        Note:
            DEPTH_CONTROLLED mode exists only for ablation studies. All
            reporter output must label results from this mode clearly.

        Args:
            depth: Fixed search depth limit per move.

        Returns:
            A MatchConfig with mode=DEPTH_CONTROLLED.
        """
        return cls(mode=MatchMode.DEPTH_CONTROLLED, fixed_depth=depth)


# ---------------------------------------------------------------------------
# Per-move statistics
# ---------------------------------------------------------------------------


@dataclass
class MoveStats:
    """Instrumentation data captured for a single move.

    Attributes:
        move_number: Sequential position of the move in the game.
        player: The player who made this move.
        move: The (row, col) cell chosen.
        nodes_visited: Search nodes expanded by the agent.
        max_depth_reached: Deepest ply searched.
        prunings: Number of pruning events.
        pruning_rate: Fraction of nodes pruned (prunings / nodes_visited).
        time_taken_ms: Wall-clock time for this move.
        effective_branching_factor: nodes_visited^(1/max_depth_reached).
        time_limit_exceeded: True if the agent exceeded its time budget.
    """

    move_number: int
    player: Player
    move: Move
    nodes_visited: int
    max_depth_reached: int
    prunings: int
    pruning_rate: float
    time_taken_ms: float
    effective_branching_factor: float
    time_limit_exceeded: bool


# ---------------------------------------------------------------------------
# Game record
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    """Complete record of a single game, including per-move instrumentation.

    Attributes:
        agent_x_name: Name of the agent playing as Player X.
        agent_o_name: Name of the agent playing as Player O.
        n: Board dimension.
        k: Winning line length.
        match_config: Budget configuration used for this game.
        result: Final game outcome.
        total_moves: Number of moves made before the game ended.
        move_history: Ordered list of all (row, col) moves.
        per_move_stats: Instrumentation for each individual move.
    """

    agent_x_name: str
    agent_o_name: str
    n: int
    k: int
    match_config: MatchConfig
    result: Result
    total_moves: int
    move_history: list[Move]
    per_move_stats: list[MoveStats] = field(default_factory=list)

    def get_stats_for_player(self, player_name: str) -> list[MoveStats]:
        """Return move stats for moves made by the named player.

        Args:
            player_name: Either agent_x_name or agent_o_name.

        Returns:
            Subset of per_move_stats where the associated player matches.
        """
        if player_name == self.agent_x_name:
            target_player = Player.X
        else:
            target_player = Player.O
        return [s for s in self.per_move_stats if s.player is target_player]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the game record to a JSON-compatible dictionary.

        Returns:
            A dictionary with all fields in JSON-compatible types.
        """
        return {
            "agent_x_name": self.agent_x_name,
            "agent_o_name": self.agent_o_name,
            "n": self.n,
            "k": self.k,
            "match_mode": self.match_config.mode.name,
            "result": self.result.name,
            "total_moves": self.total_moves,
            "move_history": list(self.move_history),
            "per_move_stats": [
                {
                    "move_number": s.move_number,
                    "player": s.player.name,
                    "move": s.move,
                    "nodes_visited": s.nodes_visited,
                    "max_depth_reached": s.max_depth_reached,
                    "prunings": s.prunings,
                    "pruning_rate": s.pruning_rate,
                    "time_taken_ms": s.time_taken_ms,
                    "effective_branching_factor": s.effective_branching_factor,
                    "time_limit_exceeded": s.time_limit_exceeded,
                }
                for s in self.per_move_stats
            ],
        }


# ---------------------------------------------------------------------------
# Duel result
# ---------------------------------------------------------------------------


@dataclass
class DuelResult:
    """Summary of a series of games between two agents.

    Attributes:
        agent_a_name: Name of agent A (first argument to Arena.duel).
        agent_b_name: Name of agent B (second argument to Arena.duel).
        agent_a_wins: Number of games won by agent A.
        agent_b_wins: Number of games won by agent B.
        draws: Number of drawn games.
        total_games: Total games played.
        n: Board dimension.
        k: Winning line length.
        match_config: Budget configuration used for all games.
        avg_nodes_a: Mean nodes visited per move by agent A.
        avg_nodes_b: Mean nodes visited per move by agent B.
        std_nodes_a: Standard deviation of nodes visited per move by A.
        std_nodes_b: Standard deviation of nodes visited per move by B.
        avg_time_ms_a: Mean wall-clock time per move for agent A.
        avg_time_ms_b: Mean wall-clock time per move for agent B.
        avg_depth_a: Mean max depth reached per move by agent A.
        avg_depth_b: Mean max depth reached per move by agent B.
        avg_ebf_a: Mean effective branching factor for agent A.
        avg_ebf_b: Mean effective branching factor for agent B.
        avg_pruning_rate_a: Mean pruning rate for agent A.
        avg_pruning_rate_b: Mean pruning rate for agent B.
        elo_a: Computed Elo rating for agent A after this duel.
        elo_b: Computed Elo rating for agent B after this duel.
        games: All individual GameRecord objects from the duel.
    """

    agent_a_name: str
    agent_b_name: str
    agent_a_wins: int
    agent_b_wins: int
    draws: int
    total_games: int
    n: int
    k: int
    match_config: MatchConfig
    avg_nodes_a: float = 0.0
    avg_nodes_b: float = 0.0
    std_nodes_a: float = 0.0
    std_nodes_b: float = 0.0
    avg_time_ms_a: float = 0.0
    avg_time_ms_b: float = 0.0
    avg_depth_a: float = 0.0
    avg_depth_b: float = 0.0
    avg_ebf_a: float = 0.0
    avg_ebf_b: float = 0.0
    avg_pruning_rate_a: float = 0.0
    avg_pruning_rate_b: float = 0.0
    elo_a: float = 1500.0
    elo_b: float = 1500.0
    games: list[GameRecord] = field(default_factory=list)

    def win_rate_a(self) -> float:
        """Fraction of games won by agent A.

        Returns:
            agent_a_wins / total_games, or 0.0 if total_games is zero.
        """
        if self.total_games == 0:
            return 0.0
        return self.agent_a_wins / self.total_games

    def win_rate_b(self) -> float:
        """Fraction of games won by agent B.

        Returns:
            agent_b_wins / total_games, or 0.0 if total_games is zero.
        """
        if self.total_games == 0:
            return 0.0
        return self.agent_b_wins / self.total_games

    def to_dict(self) -> dict[str, Any]:
        """Serialise the duel result to a JSON-compatible dictionary.

        Returns:
            A dictionary with all scalar fields and a list of game records.
        """
        return {
            "agent_a_name": self.agent_a_name,
            "agent_b_name": self.agent_b_name,
            "agent_a_wins": self.agent_a_wins,
            "agent_b_wins": self.agent_b_wins,
            "draws": self.draws,
            "total_games": self.total_games,
            "win_rate_a": self.win_rate_a(),
            "win_rate_b": self.win_rate_b(),
            "n": self.n,
            "k": self.k,
            "match_mode": self.match_config.mode.name,
            "avg_nodes_a": self.avg_nodes_a,
            "avg_nodes_b": self.avg_nodes_b,
            "std_nodes_a": self.std_nodes_a,
            "std_nodes_b": self.std_nodes_b,
            "avg_time_ms_a": self.avg_time_ms_a,
            "avg_time_ms_b": self.avg_time_ms_b,
            "avg_depth_a": self.avg_depth_a,
            "avg_depth_b": self.avg_depth_b,
            "avg_ebf_a": self.avg_ebf_a,
            "avg_ebf_b": self.avg_ebf_b,
            "avg_pruning_rate_a": self.avg_pruning_rate_a,
            "avg_pruning_rate_b": self.avg_pruning_rate_b,
            "elo_a": self.elo_a,
            "elo_b": self.elo_b,
            "games": [g.to_dict() for g in self.games],
        }


# ---------------------------------------------------------------------------
# Scalability record
# ---------------------------------------------------------------------------


@dataclass
class ScalabilityRecord:
    """Per-agent performance data collected across multiple board sizes.

    Attributes:
        agent_name: Name of the agent being profiled.
        board_sizes: Ordered list of board dimensions tested, e.g. [3, 5, 7, 9].
        match_config: Budget configuration used for all sizes.
        avg_nodes_per_size: Mean nodes visited per move at each board size.
        avg_depth_per_size: Mean max depth reached per move at each board size.
        avg_ebf_per_size: Mean effective branching factor at each board size.
        avg_time_ms_per_size: Mean wall-clock time per move at each board size.
        win_rate_per_size: Win rate vs RandomAgent at each board size.
    """

    agent_name: str
    board_sizes: list[int]
    match_config: MatchConfig
    avg_nodes_per_size: list[float] = field(default_factory=list)
    avg_depth_per_size: list[float] = field(default_factory=list)
    avg_ebf_per_size: list[float] = field(default_factory=list)
    avg_time_ms_per_size: list[float] = field(default_factory=list)
    win_rate_per_size: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Round-robin result
# ---------------------------------------------------------------------------


@dataclass
class RoundRobinResult:
    """Aggregated results from a round-robin tournament.

    Attributes:
        agents: Names of all participants.
        n: Board dimension.
        match_config: Budget configuration used for all duels.
        duel_matrix: Mapping from (agent_a_name, agent_b_name) to DuelResult.
        final_elo_ranking: Agents sorted by descending Elo, as
            [(agent_name, elo_rating), ...].
    """

    agents: list[str]
    n: int
    match_config: MatchConfig
    duel_matrix: dict[tuple[str, str], DuelResult] = field(default_factory=dict)
    final_elo_ranking: list[tuple[str, float]] = field(default_factory=list)

    def get_win_rate_matrix(self) -> dict[tuple[str, str], float]:
        """Build a win-rate lookup from every head-to-head duel.

        Returns:
            Mapping from (agent_a_name, agent_b_name) to agent_a's win rate.
        """
        return {
            pair: duel.win_rate_a()
            for pair, duel in self.duel_matrix.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the round-robin result to a JSON-compatible dictionary.

        Returns:
            A dictionary with agent names, Elo ranking, and all duel data.
        """
        return {
            "agents": self.agents,
            "n": self.n,
            "match_mode": self.match_config.mode.name,
            "final_elo_ranking": [
                {"agent": name, "elo": elo}
                for name, elo in self.final_elo_ranking
            ],
            "duels": {
                f"{a} vs {b}": duel.to_dict()
                for (a, b), duel in self.duel_matrix.items()
            },
        }


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------


def compute_elo(
    results: list[DuelResult],
    k_factor: float = 32.0,
    initial_rating: float = 1500.0,
) -> dict[str, float]:
    """Compute Elo ratings from a sequence of duel results.

    Standard Elo update formula applied game-by-game. Agents not yet seen
    start at initial_rating. Results are processed in list order.

    Args:
        results: Ordered list of DuelResult objects to process.
        k_factor: Elo K-factor controlling rating sensitivity.
        initial_rating: Starting rating for every agent.

    Returns:
        A dictionary mapping agent name to final Elo rating.
    """
    ratings: dict[str, float] = {}

    for duel in results:
        rating_a = ratings.setdefault(duel.agent_a_name, initial_rating)
        rating_b = ratings.setdefault(duel.agent_b_name, initial_rating)

        for game in duel.games:
            score_a, score_b = _game_scores(game, duel.agent_a_name)
            expected_a = _expected_score(rating_a, rating_b)
            expected_b = 1.0 - expected_a

            rating_a += k_factor * (score_a - expected_a)
            rating_b += k_factor * (score_b - expected_b)

        ratings[duel.agent_a_name] = rating_a
        ratings[duel.agent_b_name] = rating_b

    return ratings


def _game_scores(game: GameRecord, agent_a_name: str) -> tuple[float, float]:
    """Derive Elo game scores from a GameRecord.

    Args:
        game: The game record to score.
        agent_a_name: Name of the player treated as agent A.

    Returns:
        A (score_a, score_b) tuple where each score is 1.0 (win),
        0.5 (draw), or 0.0 (loss).
    """
    if game.result is Result.DRAW:
        return 0.5, 0.5

    a_plays_x = game.agent_x_name == agent_a_name
    a_won = (
        (a_plays_x and game.result is Result.X_WINS)
        or (not a_plays_x and game.result is Result.O_WINS)
    )

    return (1.0, 0.0) if a_won else (0.0, 1.0)


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Compute the expected Elo score for agent A given both ratings.

    Args:
        rating_a: Current Elo rating of agent A.
        rating_b: Current Elo rating of agent B.

    Returns:
        Expected score for A in [0, 1].
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def aggregate_move_stats(stats: list[MoveStats]) -> dict[str, float]:
    """Compute summary statistics over a list of MoveStats entries.

    Args:
        stats: The move stats to aggregate. May be empty.

    Returns:
        A dictionary with keys: mean_nodes, std_nodes, mean_time_ms,
        std_time_ms, mean_prunings, std_prunings, mean_ebf, std_ebf,
        mean_depth, std_depth. All values are 0.0 for an empty list.
    """
    if not stats:
        zero_keys = [
            "mean_nodes", "std_nodes", "mean_time_ms", "std_time_ms",
            "mean_prunings", "std_prunings", "mean_ebf", "std_ebf",
            "mean_depth", "std_depth",
        ]
        return {key: 0.0 for key in zero_keys}

    def mean_and_std(values: list[float]) -> tuple[float, float]:
        mean = statistics.mean(values) if values else 0.0
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    nodes = [float(s.nodes_visited) for s in stats]
    times = [s.time_taken_ms for s in stats]
    prunings = [float(s.prunings) for s in stats]
    ebfs = [s.effective_branching_factor for s in stats]
    depths = [float(s.max_depth_reached) for s in stats]

    mean_n, std_n = mean_and_std(nodes)
    mean_t, std_t = mean_and_std(times)
    mean_p, std_p = mean_and_std(prunings)
    mean_e, std_e = mean_and_std(ebfs)
    mean_d, std_d = mean_and_std(depths)

    return {
        "mean_nodes": mean_n,
        "std_nodes": std_n,
        "mean_time_ms": mean_t,
        "std_time_ms": std_t,
        "mean_prunings": mean_p,
        "std_prunings": std_p,
        "mean_ebf": mean_e,
        "std_ebf": std_e,
        "mean_depth": mean_d,
        "std_depth": std_d,
    }


def compute_pruning_rate(stats: list[MoveStats]) -> float:
    """Compute the average pruning rate across a list of MoveStats.

    Pruning rate = prunings / nodes_visited for each move. Entries where
    nodes_visited is zero are excluded from the average.

    Args:
        stats: Move stats to average over.

    Returns:
        Mean pruning rate in [0, 1], or 0.0 if all entries have zero nodes.
    """
    rates = [
        s.prunings / s.nodes_visited
        for s in stats
        if s.nodes_visited > 0
    ]
    if not rates:
        return 0.0
    return sum(rates) / len(rates)
