import argparse
import time
from typing import Callable

from src.agents.base_agent import BaseAgent
from src.agents.bns_agent import BNSAgent
from src.agents.bns_id_agent import BNSIDAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.minimax_alphabeta_agent import MinimaxAlphaBetaAgent
from src.agents.minimax_rewards_alphabeta_agent import MinimaxRewardsAlphaBetaAgent
from src.agents.mc_informed_agent import MCInformedAgent
from src.agents.mc_puct_agent import MCPUCTAgent
from src.agents.mc_pure_agent import MCPureAgent
from src.agents.mc_rave_agent import MCRaveAgent
from src.agents.mc_uct_agent import MCUCTAgent
from src.agents.mtdf_agent import MTDfAgent
from src.agents.mtdf_id_agent import MTDfIDAgent
from src.agents.negamax_agent import NegamaxAgent
from src.agents.negascout_agent import NegascoutAgent
from src.agents.random_agent import RandomAgent
from src.core.state import State
from src.core.tt_state import TTState
from src.core.types import Player

_TT_AGENTS = {"mtdf", "mtdf_id", "bns_id"}

AgentFactory = Callable[[str, int], BaseAgent]

AGENTS: dict[str, AgentFactory] = {
    # "random": lambda p, d: RandomAgent(),
    "minimax_ab": lambda p, d: MinimaxAlphaBetaAgent(Player[p]),
    # "minimax_rewards_ab": lambda p, d: MinimaxRewardsAlphaBetaAgent(Player[p], d),
    "negamax": lambda p, d: NegamaxAgent(d),
    "mtdf": lambda p, d: MTDfAgent(d),
    "mtdf_id": lambda p, d: MTDfIDAgent(d),
    # "negascout": lambda p, d: NegascoutAgent(d),
    # "bns": lambda p, d: BNSAgent(d),
    # "bns_id": lambda p, d: BNSIDAgent(d),
    "mc_pure": lambda p, d: MCPureAgent(n_simulations=d * 200, seed=42),
    "mc_uct": lambda p, d: MCUCTAgent(n_simulations=d * 200, seed=42),
    "mc_informed": lambda p, d: MCInformedAgent(n_simulations=d * 200, seed=42),
    "mc_rave": lambda p, d: MCRaveAgent(n_simulations=d * 200, seed=42),
    "mc_puct": lambda p, d: MCPUCTAgent(n_simulations=d * 200, seed=42),
}

class Main:
    """Orchestrates a self-play game between two AI agents."""

    _TIMEOUT = 60 * 60  # 60 minutes

    def __init__(self, n: int, k: int, agent: str, verbose: bool) -> None:
        """Initialises the game with two agents sharing the same algorithm.

        Args:
            n: Board size (n x n).
            k: Number of pieces in a row required to win.
            agent: Key into AGENTS selecting the algorithm.
            verbose: Whether to print the board after each move.
        """
        self._n = n
        self._k = k
        max_depth = n * n
        self._agents: dict[str, BaseAgent] = {
            p: AGENTS[agent](p, max_depth) for p in ("X", "O")
        }
        self._state = TTState(n=n, k=k) if agent in _TT_AGENTS else State(n=n, k=k)
        self._verbose = verbose

    def _algo_name(self) -> str:
        """Returns the class name of the X agent as the algorithm label."""
        return type(self._agents["X"]).__name__

    def _print_result(self, winner_label: str, elapsed: float) -> None:
        """Prints the game result in a standard one-line format.

        Args:
            winner_label: Display string for the winner (e.g. "X", "Draw", "Timeout").
            elapsed: Wall-clock seconds the game took.
        """
        print(
            f"Winner: {winner_label} "
            f"| n={self._n} k={self._k} "
            f"| Algorithm: {self._algo_name()} "
            f"| States visited: {self._state.state_count} "
            f"| Time: {elapsed:.2f}s"
        )

    def _play_move(self) -> None:
        """Asks the current player's agent to act and applies the move to state."""
        player = self._state.current_player
        row, col = self._agents[player.name].act(self._state)
        self._state.apply(row, col)
        if self._verbose:
            print(f"Player {player.name} plays ({row}, {col})")
            print(self._state.board.render())

    def _game_loop(self, deadline: float) -> str:
        """Plays moves until terminal state or the deadline passes.

        Args:
            deadline: time.time() value after which play stops early.

        Returns:
            Winner label: player name, "Draw", or "Timeout".
        """
        # ponytail: checked between moves, not inside agent.act() — a hung
        # search still runs to completion. Cross-platform interruption mid-move
        # needs a worker thread/process; add if a single move can itself hang.
        while not self._state.is_terminal():
            if time.time() >= deadline:
                return "Timeout"
            self._play_move()
        winner = self._state.winner()
        return winner.name if winner else "Draw"

    def run(self) -> None:
        """Runs the game with a soft wall-clock timeout."""
        start = time.time()
        winner_label = self._game_loop(deadline=start + self._TIMEOUT)
        self._print_result(winner_label, time.time() - start)


def _parse_args() -> argparse.Namespace:
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, required=True, help="Board size (n x n)")
    parser.add_argument(
        "-k", type=int, required=True, help="Win condition (k in a row)"
    )
    parser.add_argument(
        "-agt",
        choices=list(AGENTS),
        required=True,
        help=f"AI agent. Choices: {', '.join(AGENTS)}",
    )
    parser.add_argument("-v", action="store_true", help="Print board after each move")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    Main(n=args.n, k=args.k, agent=args.agt, verbose=args.v).run()
