"""Entry point for the n×n Tic-Tac-Toe framework.

Provides a CLI with five modes:
  human       — Human vs Human interactive game.
  demo        — HumanAgent vs RandomAgent on a 3×3 board.
  sanity      — Run the BruteForceOracle vs RandomAgent sanity check.
  benchmark   — Placeholder message (no algorithm agents loaded yet).
  correctness — Run verify_agent_on_known_positions on BruteForceOracle.

Usage examples:
  python -m tictactoe.main --mode demo
  python -m tictactoe.main --mode sanity --games 20
  python -m tictactoe.main --mode correctness
  python -m tictactoe.main --mode human --n 5
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        A fully configured ArgumentParser.
    """
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
        help="Winning line length. Defaults to n.",
    )
    parser.add_argument(
        "--games", type=int, default=100,
        help="Number of games for benchmark / sanity modes.",
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
    return parser


def build_match_config(args: argparse.Namespace):
    """Build a MatchConfig from parsed CLI arguments.

    Args:
        args: Parsed argument namespace.

    Returns:
        A MatchConfig appropriate for the selected match mode.
    """
    from tictactoe.benchmark.metrics import MatchConfig

    if args.match_mode == "node":
        return MatchConfig.node_controlled(args.node_budget)
    if args.match_mode == "depth":
        return MatchConfig.depth_controlled(args.fixed_depth)
    return MatchConfig.time_controlled(args.time_limit_ms)


def run_human(args: argparse.Namespace) -> None:
    """Run a Human vs Human interactive game.

    Args:
        args: Parsed CLI arguments.
    """
    from tictactoe.agents.human_agent import HumanAgent
    from tictactoe.core.game import Game

    n = args.n
    k = args.k if args.k is not None else n
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
    """Run a Human vs RandomAgent demo game on a 3×3 board.

    Args:
        args: Parsed CLI arguments.
    """
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
    """Run the BruteForceOracle vs RandomAgent sanity check.

    Args:
        args: Parsed CLI arguments.
    """
    from tictactoe.benchmark.arena import Arena
    from tictactoe.benchmark.correctness import BruteForceOracle
    from tictactoe.benchmark.reporter import print_correctness_report

    games = args.games if args.games % 2 == 0 else args.games + 1
    print(f"\nSanity check: BruteForceOracle vs RandomAgent ({games} games on 3×3)\n")

    arena = Arena(n=3, k=3, num_games=games, match_config=build_match_config(args))
    arena.set_seed(args.seed)
    result = arena.sanity_check(BruteForceOracle(), games=games)

    status = "PASSED" if result["passed"] else "FAILED"
    print(f"  Status   : {status}")
    print(f"  Win rate : {result['win_rate']:.3f}")
    print(f"  Agent    : {result['agent_name']}")


def run_benchmark(_args: argparse.Namespace) -> None:
    """Print a placeholder message for benchmark mode.

    Args:
        _args: Parsed CLI arguments (unused).
    """
    print(
        "\nNo algorithm agents loaded yet. "
        "Add agents to tictactoe/agents/ and register them here to run benchmarks.\n"
    )


def run_correctness(args: argparse.Namespace) -> None:
    """Run correctness verification on BruteForceOracle against all known positions.

    Args:
        args: Parsed CLI arguments.
    """
    from tictactoe.benchmark.correctness import BruteForceOracle, verify_agent_on_known_positions
    from tictactoe.benchmark.reporter import print_correctness_report

    print("\nCorrectness check: BruteForceOracle on KNOWN_POSITIONS\n")
    oracle = BruteForceOracle()
    report = verify_agent_on_known_positions(oracle)
    print_correctness_report(report, oracle.get_name())


def _resolve_config_path(raw: str) -> pathlib.Path:
    """Return an absolute path to the config file.

    Relative paths are resolved against the project root (the directory that
    contains the tictactoe/ package directory).

    Args:
        raw: The raw --config argument value.

    Returns:
        Absolute Path to the config file.
    """
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

    # Load and validate config before any agents are instantiated.
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

    handler = handlers[args.mode]
    handler(args)


if __name__ == "__main__":
    main()
