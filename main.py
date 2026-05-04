import argparse
import time
from typing import Callable

from src.agents.base_agent import BaseAgent
from src.agents.bns_agent import BNSAgent
from src.agents.bns_id_agent import BNSIDAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.minimax_alphabeta_agent import MinimaxAlphaBetaAgent
from src.agents.minimax_rewards_alphabeta_agent import MinimaxRewardsAlphaBetaAgent
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
    "random": lambda p, d: RandomAgent(),
    "minimax": lambda p, d: MinimaxAgent(Player[p]),
    "minimax_ab": lambda p, d: MinimaxAlphaBetaAgent(Player[p]),
    "minimax_rewards_ab": lambda p, d: MinimaxRewardsAlphaBetaAgent(Player[p], d),
    "negamax": lambda p, d: NegamaxAgent(d),
    "mtdf": lambda p, d: MTDfAgent(d),
    "mtdf_id": lambda p, d: MTDfIDAgent(d),
    "negascout": lambda p, d: NegascoutAgent(d),
    "bns": lambda p, d: BNSAgent(d),
    "bns_id": lambda p, d: BNSIDAgent(d),
}


class Main:
    def __init__(self, n: int, k: int, agent: str, verbose: bool) -> None:
        self._n = n
        self._k = k
        max_depth = n * n
        self._agents: dict[str, BaseAgent] = {
            p: AGENTS[agent](p, max_depth) for p in ("X", "O")
        }
        self._state = TTState(n=n, k=k) if agent in _TT_AGENTS else State(n=n, k=k)
        self._verbose = verbose

    def run(self) -> None:
        start = time.time()
        while not self._state.is_terminal():
            player = self._state.current_player
            row, col = self._agents[player.name].act(self._state)
            self._state.apply(row, col)
            if self._verbose:
                print(f"Player {player.name} plays ({row}, {col})")
                print(self._state.board.render())
        elapsed = time.time() - start

        winner = self._state.winner()
        algo = type(self._agents["X"]).__name__
        print(
            f"Winner: {winner.name if winner else 'Draw'} "
            f"| n={self._n} k={self._k} "
            f"| Algorithm: {algo} "
            f"| States visited: {self._state.state_count} "
            f"| Time: {elapsed:.2f}s"
        )


def _parse_args() -> argparse.Namespace:
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
