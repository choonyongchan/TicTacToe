"""Results formatting and export for benchmark runs.

All console output in the framework routes through this module. CSV and JSON
export functions write structured data to disk. Plotting functions require
matplotlib and silently skip when it is unavailable.
"""

from __future__ import annotations

import csv
import json
from typing import Any

from tictactoe.benchmark.metrics import (
    DuelResult,
    GameRecord,
    RoundRobinResult,
    ScalabilityRecord,
)
from tictactoe.core.types import MatchMode, Result


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------


def print_duel_summary(result: DuelResult) -> None:
    """Print a concise summary table for a head-to-head duel.

    Displays wins/draws/losses, Elo ratings, win rates, and average
    efficiency metrics (nodes, time, EBF, depth, pruning rate).

    Args:
        result: The duel result to display.
    """
    mode_label = _mode_label(result.match_config.mode)
    print(f"\n{'='*60}")
    print(f"  Duel: {result.agent_a_name}  vs  {result.agent_b_name}")
    print(f"  Board: {result.n}×{result.n}  k={result.k}  Mode: {mode_label}")
    print(f"{'='*60}")
    print(f"  {'Metric':<28} {'Agent A':>12} {'Agent B':>12}")
    print(f"  {'-'*52}")

    rows = [
        ("Wins",          f"{result.agent_a_wins}",     f"{result.agent_b_wins}"),
        ("Draws",         f"{result.draws}",             f"{result.draws}"),
        ("Total Games",   f"{result.total_games}",       f"{result.total_games}"),
        ("Win Rate",      f"{result.win_rate_a():.3f}",  f"{result.win_rate_b():.3f}"),
        ("Elo",           f"{result.elo_a:.1f}",         f"{result.elo_b:.1f}"),
        ("Avg Nodes",     f"{result.avg_nodes_a:.1f}",   f"{result.avg_nodes_b:.1f}"),
        ("Avg Time (ms)", f"{result.avg_time_ms_a:.2f}", f"{result.avg_time_ms_b:.2f}"),
        ("Avg EBF",       f"{result.avg_ebf_a:.3f}",     f"{result.avg_ebf_b:.3f}"),
        ("Avg Depth",     f"{result.avg_depth_a:.1f}",   f"{result.avg_depth_b:.1f}"),
        ("Avg Prune Rate",f"{result.avg_pruning_rate_a:.3f}", f"{result.avg_pruning_rate_b:.3f}"),
    ]

    for label, val_a, val_b in rows:
        print(f"  {label:<28} {val_a:>12} {val_b:>12}")

    agent_label = f"  Winner: {result.agent_a_name}" if result.win_rate_a() > 0.5 else (
        f"  Winner: {result.agent_b_name}" if result.win_rate_b() > 0.5 else "  Result: Draw series"
    )
    print(agent_label)
    print(f"{'='*60}\n")


def print_efficiency_comparison(results: list[DuelResult]) -> None:
    """Print an efficiency-focused comparison table for Tier 1 algorithms.

    Ranks agents by average nodes (ascending). Win rate is excluded from
    this table because exact-search algorithms at equal depth always draw.

    Args:
        results: DuelResult objects whose agents are compared.
    """
    # Collect per-agent efficiency data across all duels.
    agent_data: dict[str, dict[str, float]] = {}

    for duel in results:
        _merge_agent_stats(agent_data, duel.agent_a_name, duel, side="a")
        _merge_agent_stats(agent_data, duel.agent_b_name, duel, side="b")

    sorted_agents = sorted(agent_data.items(), key=lambda kv: kv[1]["avg_nodes"])

    print(f"\n{'='*90}")
    print("  Efficiency Comparison (Tier 1 — ranked by avg nodes, lower is better)")
    print(f"{'='*90}")
    header = f"  {'Agent':<28} {'Avg Nodes':>12} {'Std Nodes':>12} {'Avg Time':>10} {'Avg Depth':>10} {'Avg EBF':>10} {'Prune Rate':>12}"
    print(header)
    print(f"  {'-'*86}")

    for agent_name, data in sorted_agents:
        print(
            f"  {agent_name:<28} "
            f"{data['avg_nodes']:>12.1f} "
            f"{data['std_nodes']:>12.1f} "
            f"{data['avg_time_ms']:>10.2f} "
            f"{data['avg_depth']:>10.1f} "
            f"{data['avg_ebf']:>10.3f} "
            f"{data['avg_pruning_rate']:>12.3f}"
        )

    print(f"{'='*90}\n")


def print_round_robin_table(result: RoundRobinResult) -> None:
    """Print a win-rate matrix and Elo ranking for a round-robin tournament.

    Args:
        result: The round-robin result to display.
    """
    agents = result.agents
    win_rates = result.get_win_rate_matrix()

    print(f"\n{'='*70}")
    print(f"  Round-Robin Tournament  (n={result.n})")
    print(f"{'='*70}")

    col_width = 10
    header = f"  {'':20}" + "".join(f"{name[:9]:>{col_width}}" for name in agents)
    print(header)
    print(f"  {'-'*68}")

    for row_agent in agents:
        row = f"  {row_agent[:19]:<20}"
        for col_agent in agents:
            if row_agent == col_agent:
                row += f"{'---':>{col_width}}"
            else:
                rate = win_rates.get((row_agent, col_agent), 0.0)
                row += f"{rate:>{col_width}.3f}"
        print(row)

    print(f"\n  Elo Ranking:")
    print(f"  {'-'*40}")
    for rank, (agent_name, elo) in enumerate(result.final_elo_ranking, start=1):
        print(f"  {rank}. {agent_name:<30} {elo:.1f}")
    print(f"{'='*70}\n")


def print_scalability_table(records: list[ScalabilityRecord]) -> None:
    """Print a per-agent scalability summary across board sizes.

    Each cell shows avg_depth / avg_ebf for that (agent, board_size) pair.

    Args:
        records: Scalability records from Arena.scalability_sweep.
    """
    if not records:
        print("No scalability records to display.")
        return

    board_sizes = records[0].board_sizes
    print(f"\n{'='*80}")
    print("  Scalability Summary  (depth / EBF per board size)")
    print(f"{'='*80}")

    col_w = 14
    header = f"  {'Agent':<24}" + "".join(f"  n={n:<{col_w-4}}" for n in board_sizes)
    print(header)
    print(f"  {'-'*78}")

    for record in records:
        row = f"  {record.agent_name[:23]:<24}"
        for depth, ebf in zip(record.avg_depth_per_size, record.avg_ebf_per_size):
            cell = f"{depth:.1f}/{ebf:.2f}"
            row += f"  {cell:<{col_w-2}}"
        print(row)

    print(f"{'='*80}\n")


def print_correctness_report(report: dict, agent_name: str) -> None:
    """Print a pass/fail correctness report for a single agent.

    Args:
        report: The dict returned by verify_agent_on_known_positions or
            verify_oracle_never_loses.
        agent_name: Name of the agent being reported on.
    """
    passed = report.get("passed", False)
    status = "PASSED" if passed else "FAILED"
    print(f"\n  Correctness Report — {agent_name}: {status}")
    print(f"  {'-'*50}")

    if "total" in report:
        print(f"  Positions tested : {report['total']}")
        print(f"  Correct          : {report['correct']}")
        failures = report.get("failures", [])
        if failures:
            print(f"  Failures ({len(failures)}):")
            for failure in failures:
                print(f"    - {failure.get('description', 'unknown')}")

    if "agent_losses" in report:
        print(f"  Agent losses     : {report['agent_losses']}")
        print(f"  Draws            : {report['draws']}")
        print(f"  Agent wins       : {report['agent_wins']}")
        print(f"  Games played     : {report['games']}")

    print()


# ---------------------------------------------------------------------------
# CSV / JSON export
# ---------------------------------------------------------------------------


def export_to_csv(result: RoundRobinResult, path: str) -> None:
    """Export round-robin win rates to a CSV file.

    Columns: agent_a, agent_b, agent_a_wins, agent_b_wins, draws,
    total_games, win_rate_a, elo_a, elo_b.

    Args:
        result: The round-robin result to export.
        path: File path for the output CSV.
    """
    fieldnames = [
        "agent_a", "agent_b", "agent_a_wins", "agent_b_wins",
        "draws", "total_games", "win_rate_a", "elo_a", "elo_b",
    ]
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for duel in result.duel_matrix.values():
            writer.writerow({
                "agent_a": duel.agent_a_name,
                "agent_b": duel.agent_b_name,
                "agent_a_wins": duel.agent_a_wins,
                "agent_b_wins": duel.agent_b_wins,
                "draws": duel.draws,
                "total_games": duel.total_games,
                "win_rate_a": f"{duel.win_rate_a():.4f}",
                "elo_a": f"{duel.elo_a:.1f}",
                "elo_b": f"{duel.elo_b:.1f}",
            })


def export_to_json(result: RoundRobinResult, path: str) -> None:
    """Export a round-robin result to a JSON file.

    Args:
        result: The round-robin result to export.
        path: File path for the output JSON.
    """
    with open(path, "w") as json_file:
        json.dump(result.to_dict(), json_file, indent=2)


def export_scalability_csv(
    records: list[ScalabilityRecord], path: str
) -> None:
    """Export scalability data to a CSV file (one row per agent × board size).

    Columns: agent, n, avg_nodes, avg_depth, avg_ebf, avg_time_ms, win_rate.

    Args:
        records: Scalability records from Arena.scalability_sweep.
        path: File path for the output CSV.
    """
    fieldnames = ["agent", "n", "avg_nodes", "avg_depth", "avg_ebf", "avg_time_ms", "win_rate"]
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            for i, board_size in enumerate(record.board_sizes):
                writer.writerow({
                    "agent": record.agent_name,
                    "n": board_size,
                    "avg_nodes": f"{record.avg_nodes_per_size[i]:.2f}",
                    "avg_depth": f"{record.avg_depth_per_size[i]:.2f}",
                    "avg_ebf": f"{record.avg_ebf_per_size[i]:.4f}",
                    "avg_time_ms": f"{record.avg_time_ms_per_size[i]:.2f}",
                    "win_rate": f"{record.win_rate_per_size[i]:.4f}",
                })


# ---------------------------------------------------------------------------
# Optional matplotlib plots
# ---------------------------------------------------------------------------


def plot_scalability(
    records: list[ScalabilityRecord], metric: str = "avg_depth"
) -> None:
    """Plot a scaling curve: x=board_size, y=metric, one line per agent.

    Silently skips if matplotlib is not installed.

    Args:
        records: Scalability records to plot.
        metric: Which metric to plot on the y-axis. One of:
            "avg_depth", "avg_nodes", "avg_ebf", "avg_time_ms", "win_rate".
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return

    metric_to_attr = {
        "avg_depth": "avg_depth_per_size",
        "avg_nodes": "avg_nodes_per_size",
        "avg_ebf": "avg_ebf_per_size",
        "avg_time_ms": "avg_time_ms_per_size",
        "win_rate": "win_rate_per_size",
    }
    attr_name = metric_to_attr.get(metric, "avg_depth_per_size")

    fig, ax = plt.subplots()
    for record in records:
        y_values = getattr(record, attr_name, [])
        ax.plot(record.board_sizes, y_values, marker="o", label=record.agent_name)

    ax.set_xlabel("Board size n")
    ax.set_ylabel(metric)
    ax.set_title(f"Scalability — {metric}")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_elo_progression(
    records: list[GameRecord], agent_names: list[str]
) -> None:
    """Plot Elo rating progression over a sequence of games.

    Silently skips if matplotlib is not installed.

    Args:
        records: Ordered list of GameRecord objects (one per game).
        agent_names: The agents whose Elo trajectories to plot.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return

    # Build an Elo trajectory by replaying games one-by-one.
    ratings_over_time: dict[str, list[float]] = {name: [1500.0] for name in agent_names}

    current_ratings: dict[str, float] = {name: 1500.0 for name in agent_names}

    for game in records:
        if game.agent_x_name not in agent_names or game.agent_o_name not in agent_names:
            continue

        if game.result is Result.DRAW:
            score_x, score_o = 0.5, 0.5
        elif game.result is Result.X_WINS:
            score_x, score_o = 1.0, 0.0
        else:
            score_x, score_o = 0.0, 1.0

        rating_x = current_ratings[game.agent_x_name]
        rating_o = current_ratings[game.agent_o_name]
        exp_x = 1.0 / (1.0 + 10.0 ** ((rating_o - rating_x) / 400.0))
        exp_o = 1.0 - exp_x
        current_ratings[game.agent_x_name] += 32.0 * (score_x - exp_x)
        current_ratings[game.agent_o_name] += 32.0 * (score_o - exp_o)
        for name in agent_names:
            ratings_over_time[name].append(current_ratings[name])

    fig, ax = plt.subplots()
    for name, ratings in ratings_over_time.items():
        ax.plot(ratings, label=name)

    ax.set_xlabel("Games played")
    ax.set_ylabel("Elo rating")
    ax.set_title("Elo progression")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _mode_label(mode: MatchMode) -> str:
    """Return a human-readable label for a MatchMode.

    Args:
        mode: The match mode to label.

    Returns:
        A short descriptive string.
    """
    labels = {
        MatchMode.TIME_CONTROLLED: "Time-controlled",
        MatchMode.NODE_CONTROLLED: "Node-controlled",
        MatchMode.DEPTH_CONTROLLED: "Depth-controlled [ABLATION]",
    }
    return labels.get(mode, mode.name)


def _merge_agent_stats(
    data: dict[str, dict[str, float]],
    agent_name: str,
    duel: DuelResult,
    side: str,
) -> None:
    """Accumulate duel efficiency stats into the agent_data dict.

    Args:
        data: Accumulator dict to update in place.
        agent_name: Name of the agent.
        duel: The duel to pull stats from.
        side: Either "a" or "b" to select which agent's stats to read.
    """
    if agent_name not in data:
        data[agent_name] = {
            "avg_nodes": 0.0, "std_nodes": 0.0, "avg_time_ms": 0.0,
            "avg_depth": 0.0, "avg_ebf": 0.0, "avg_pruning_rate": 0.0,
            "_count": 0,
        }
    entry = data[agent_name]
    count = entry["_count"] + 1
    entry["_count"] = count

    def running_mean(old: float, new: float) -> float:
        return (old * (count - 1) + new) / count

    if side == "a":
        entry["avg_nodes"] = running_mean(entry["avg_nodes"], duel.avg_nodes_a)
        entry["std_nodes"] = running_mean(entry["std_nodes"], duel.std_nodes_a)
        entry["avg_time_ms"] = running_mean(entry["avg_time_ms"], duel.avg_time_ms_a)
        entry["avg_depth"] = running_mean(entry["avg_depth"], duel.avg_depth_a)
        entry["avg_ebf"] = running_mean(entry["avg_ebf"], duel.avg_ebf_a)
        entry["avg_pruning_rate"] = running_mean(entry["avg_pruning_rate"], duel.avg_pruning_rate_a)
    else:
        entry["avg_nodes"] = running_mean(entry["avg_nodes"], duel.avg_nodes_b)
        entry["std_nodes"] = running_mean(entry["std_nodes"], duel.std_nodes_b)
        entry["avg_time_ms"] = running_mean(entry["avg_time_ms"], duel.avg_time_ms_b)
        entry["avg_depth"] = running_mean(entry["avg_depth"], duel.avg_depth_b)
        entry["avg_ebf"] = running_mean(entry["avg_ebf"], duel.avg_ebf_b)
        entry["avg_pruning_rate"] = running_mean(entry["avg_pruning_rate"], duel.avg_pruning_rate_b)
