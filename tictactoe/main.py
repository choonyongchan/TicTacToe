"""Entry point for the n×n Tic-Tac-Toe framework.

Provides a CLI with five modes:
  human       — Human vs Human interactive game.
  demo        — HumanAgent vs RandomAgent on a 3×3 board.
  sanity      — Run the BruteForceOracle vs RandomAgent sanity check.
  benchmark   — Scalability sweep + round-robin across Tier 1-3 agents.
  correctness — Run verify_agent_on_known_positions on BruteForceOracle.

Usage examples:
  python -m tictactoe.main --mode demo
  python -m tictactoe.main --mode sanity --games 20
  python -m tictactoe.main --mode correctness
  python -m tictactoe.main --mode human --n 5
  python -m tictactoe.main --mode benchmark --boards 3,5,10,20 --time-limit-ms 500
  python -m tictactoe.main --mode benchmark --boards 3,5,10,20,50,100 --match-mode node --node-budget 50000
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="n×n Tic-Tac-Toe research framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["human", "demo", "sanity", "benchmark", "correctness"],
        default="demo",
        help="Which mode to run.",
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        dest="config",
        help="Path to config.toml. Relative paths are resolved from the project root.",
    )
    parser.add_argument("--n", type=int, default=3, help="Board dimension.")
    parser.add_argument(
        "--k", type=int, default=None,
        help="Winning line length. Defaults to min(n, 5).",
    )
    parser.add_argument(
        "--games", type=int, default=20,
        help="Games per pair for benchmark / sanity modes.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--verbose", action="store_true", help="Print board after each move.")
    parser.add_argument(
        "--time-limit-ms", type=float, default=1000.0,
        dest="time_limit_ms",
        help="Per-move time budget in milliseconds (TIME_CONTROLLED mode).",
    )
    parser.add_argument(
        "--match-mode",
        choices=["time", "node", "depth"],
        default="time",
        dest="match_mode",
        help="Match control mode.",
    )
    parser.add_argument(
        "--node-budget", type=int, default=100_000,
        dest="node_budget",
        help="Per-move node budget (NODE_CONTROLLED mode).",
    )
    parser.add_argument(
        "--fixed-depth", type=int, default=4,
        dest="fixed_depth",
        help="Fixed search depth (DEPTH_CONTROLLED mode).",
    )
    parser.add_argument(
        "--boards",
        default=None,
        metavar="SIZES",
        help=(
            "Comma-separated board sizes for the benchmark scalability sweep, "
            "e.g. '3,5,10,20,50,100'. Defaults to the value of --n."
        ),
    )
    parser.add_argument(
        "--tiers",
        default="1,2,3",
        metavar="TIERS",
        help="Comma-separated tier numbers to include in benchmark (e.g. '1,2,3').",
    )
    return parser


def build_match_config(args: argparse.Namespace):
    from tictactoe.benchmark.metrics import MatchConfig

    if args.match_mode == "node":
        return MatchConfig.node_controlled(args.node_budget)
    if args.match_mode == "depth":
        return MatchConfig.depth_controlled(args.fixed_depth)
    return MatchConfig.time_controlled(args.time_limit_ms)


def _build_tier_agents(config) -> dict[int, list]:
    """Instantiate all benchmarkable (Tier 1–3) agents.

    Tier 4 agents (RL) require pre-trained models and are excluded from
    automated benchmarks. Load them manually and pass them to Arena.duel.

    Args:
        config: MatchConfig to pass to each agent.

    Returns:
        Dict mapping tier number → list of agents for that tier.
    """
    from tictactoe.agents.classic_search.minimax_ab import MinimaxAB
    from tictactoe.agents.classic_search.negamax import NegaMax
    from tictactoe.agents.classic_search.negascout import NegaScout
    from tictactoe.agents.classic_search.mtdf import MTDf
    from tictactoe.agents.heuristic_search.minimax_ab_enhanced import MinimaxABEnhanced
    from tictactoe.agents.heuristic_search.negamax_enhanced import NegaMaxEnhanced
    from tictactoe.agents.heuristic_search.negascout_enhanced import NegaScoutEnhanced
    from tictactoe.agents.heuristic_search.mtdf_enhanced import MTDfEnhanced
    from tictactoe.agents.monte_carlo.mcts_vanilla import MCTSVanilla
    from tictactoe.agents.monte_carlo.mcts_rave import MCTSRave
    from tictactoe.agents.monte_carlo.mcts_heuristic_rollout import MCTSHeuristicRollout

    return {
        1: [
            MinimaxAB(match_config=config),
            NegaMax(match_config=config),
            NegaScout(match_config=config),
            MTDf(match_config=config),
        ],
        2: [
            MinimaxABEnhanced(match_config=config),
            NegaMaxEnhanced(match_config=config),
            NegaScoutEnhanced(match_config=config),
            MTDfEnhanced(match_config=config),
        ],
        3: [
            MCTSVanilla(match_config=config),
            MCTSRave(match_config=config),
            MCTSHeuristicRollout(match_config=config),
        ],
    }


def run_human(args: argparse.Namespace) -> None:
    from tictactoe.agents.human_agent import HumanAgent
    from tictactoe.core.game import Game

    n = args.n
    k = args.k if args.k is not None else min(n, 5) if n > 5 else n
    config = build_match_config(args)

    print(f"\nHuman vs Human | {n}×{n} board | win length={k}\n")
    game = Game(
        agent_x=HumanAgent("Player 1"),
        agent_o=HumanAgent("Player 2"),
        n=n,
        k=k,
        match_config=config,
    )
    result = game.run(verbose=False)
    from tictactoe.core.board import Board
    print(Board.render(game.state.board, n, game.state.last_move))
    print(f"\nResult: {result.name}")


def run_demo(args: argparse.Namespace) -> None:
    from tictactoe.agents.human_agent import HumanAgent
    from tictactoe.agents.random_agent import RandomAgent
    from tictactoe.core.game import Game

    config = build_match_config(args)
    print("\nDemo: Human (X) vs RandomAgent (O) | 3×3\n")
    game = Game(
        agent_x=HumanAgent("Human"),
        agent_o=RandomAgent(seed=args.seed),
        n=3,
        k=3,
        match_config=config,
    )
    result = game.run(verbose=False)
    from tictactoe.core.board import Board
    print(Board.render(game.state.board, 3, game.state.last_move))
    print(f"\nResult: {result.name}")


def run_sanity(args: argparse.Namespace) -> None:
    from tictactoe.benchmark.arena import Arena
    from tictactoe.benchmark.correctness import BruteForceOracle

    games = args.games if args.games % 2 == 0 else args.games + 1
    print(f"\nSanity check: BruteForceOracle vs RandomAgent ({games} games on 3×3)\n")

    arena = Arena(n=3, k=3, num_games=games, match_config=build_match_config(args))
    arena.set_seed(args.seed)
    result = arena.sanity_check(BruteForceOracle(), games=games)

    status = "PASSED" if result["passed"] else "FAILED"
    print(f"  Status   : {status}")
    print(f"  Win rate : {result['win_rate']:.3f}")
    print(f"  Agent    : {result['agent_name']}")


def run_benchmark(args: argparse.Namespace) -> None:
    """Run scalability sweep + optional round-robin for Tier 1-3 agents.

    Board sizes come from --boards (comma-separated). When a single board
    size is given, also runs a within-tier round-robin tournament.

    The comprehensive performance report shows, per agent per board size:
    - Average nodes visited (= nodes expanded in this framework)
    - Average effective branching factor (EBF)
    - Average time per move (ms)
    - Win rate vs RandomAgent
    - Budget exhausted: how many moves hit the search budget limit
    """
    from tictactoe.benchmark.arena import Arena
    from tictactoe.benchmark.reporter import print_round_robin_table, print_performance_report

    config = build_match_config(args)

    # Parse board sizes and selected tiers.
    if args.boards:
        board_sizes = [int(x.strip()) for x in args.boards.split(",")]
    else:
        board_sizes = [args.n]

    selected_tiers = [int(t.strip()) for t in args.tiers.split(",")]

    # Primary board/k for round-robin.
    primary_n = board_sizes[0]
    primary_k = args.k if args.k is not None else min(primary_n, 5) if primary_n > 5 else primary_n

    all_tier_agents = _build_tier_agents(config)

    # Build flat list and tier mapping for the report.
    sweep_agents = []
    agent_tiers: dict[str, int] = {}
    for tier in selected_tiers:
        for agent in all_tier_agents.get(tier, []):
            sweep_agents.append(agent)
            agent_tiers[agent.get_name()] = tier

    if not sweep_agents:
        print("\nNo agents to benchmark. Check --tiers argument.")
        return

    print(f"\n{'='*60}")
    print("  Benchmark — Scalability Sweep")
    print(f"  Board sizes : {board_sizes}")
    print(f"  Tiers       : {selected_tiers}")
    print(f"  Games/size  : {args.games}  |  Seed: {args.seed}")
    print(f"  Mode        : {args.match_mode}")
    print(f"{'='*60}")

    # Scalability sweep: each agent vs RandomAgent at every board size.
    arena = Arena(
        n=primary_n,
        k=primary_k,
        num_games=args.games,
        match_config=config,
    )
    arena.set_seed(args.seed)

    records = arena.scalability_sweep(
        agents=sweep_agents,
        board_sizes=board_sizes,
        games_per_size=args.games,
        k_override=args.k,
    )

    print_performance_report(records, agent_tiers, k_override=args.k)

    # Round-robin within each tier for the primary board size only.
    if len(board_sizes) == 1:
        for tier in selected_tiers:
            tier_agents = all_tier_agents.get(tier, [])
            if len(tier_agents) < 2:
                continue
            print(f"\n{'='*60}")
            print(f"  Tier {tier} Round-Robin | {primary_n}×{primary_n} (k={primary_k})")
            print(f"  {args.games} games per pair | Mode: {args.match_mode}")
            print(f"{'='*60}")
            rr_result = arena.round_robin(tier_agents)
            print_round_robin_table(rr_result)


def run_correctness(args: argparse.Namespace) -> None:
    from tictactoe.benchmark.correctness import BruteForceOracle, verify_agent_on_known_positions
    from tictactoe.benchmark.reporter import print_correctness_report

    print("\nCorrectness check: BruteForceOracle on KNOWN_POSITIONS\n")
    oracle = BruteForceOracle()
    report = verify_agent_on_known_positions(oracle)
    print_correctness_report(report, oracle.get_name())


def _resolve_config_path(raw: str) -> pathlib.Path:
    p = pathlib.Path(raw)
    if not p.is_absolute():
        project_root = pathlib.Path(__file__).parent.parent
        p = project_root / p
    return p


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate mode handler."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = build_parser()
    args = parser.parse_args()

    from tictactoe.config import load_config, ConfigError
    config_path = _resolve_config_path(args.config)
    try:
        load_config(config_path)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    handlers = {
        "human": run_human,
        "demo": run_demo,
        "sanity": run_sanity,
        "benchmark": run_benchmark,
        "correctness": run_correctness,
    }

    handlers[args.mode](args)


if __name__ == "__main__":
    main()
