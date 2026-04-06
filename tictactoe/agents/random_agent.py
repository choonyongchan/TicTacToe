"""Random agent — the benchmark floor for all algorithmic comparison.

RandomAgent selects a uniformly random legal move on every turn. It has two
roles in the framework:

1. **Sanity check floor**: Any real algorithm must achieve
   > RANDOM_AGENT_WIN_THRESHOLD win rate against RandomAgent on n=3.
   Failure indicates a bug in the algorithm under test.

2. **Position generator**: Arena uses RandomAgent to produce diverse mid-game
   board positions for correctness testing and profiling.
"""

from __future__ import annotations

import random

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Move


class RandomAgent(BaseAgent):
    """A uniform-random legal move selector.

    Attributes:
        seed: The optional random seed for reproducibility. When provided,
            the internal RNG is seeded at construction time.
        _rng: The private Random instance used for all move selection.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialise the random agent.

        Args:
            seed: Optional integer seed for the internal random number
                generator. Passing the same seed produces the same move
                sequence against the same sequence of board states.
        """
        self.seed = seed
        self._rng = random.Random(seed)

    def choose_move(self, state: GameState) -> Move:
        """Select a uniformly random empty cell.

        Args:
            state: The current game state.

        Returns:
            A randomly chosen (row, col) from the set of empty cells.

        Raises:
            ValueError: If there are no empty cells on the board.
        """
        empty_cells = Board.get_all_empty_cells(state.board)

        if not empty_cells:
            raise ValueError("No legal moves available — board is full.")

        chosen = self._rng.choice(empty_cells)

        # Minimal instrumentation: one node visited, no search depth.
        state.nodes_visited = 1
        state.max_depth_reached = 0
        state.prunings = 0
        state.compute_ebf()

        return chosen

    def get_name(self) -> str:
        """Return the display name for this agent.

        Returns:
            The string "RandomAgent".
        """
        return f"RandomAgent(seed={self.seed})" if self.seed is not None else "RandomAgent"

    def get_tier(self) -> int:
        """Return the baseline tier for random agents.

        Returns:
            0, indicating a non-algorithmic baseline.
        """
        return 0
