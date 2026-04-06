"""Human player agent that reads moves from the console.

The HumanAgent prints the board and prompts the user to enter a (row, col)
pair. Invalid inputs are rejected with an explanation and the prompt repeats.

All instrumentation fields are set to zero because humans do not perform
algorithmic search.
"""

from __future__ import annotations

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Move


class HumanAgent(BaseAgent):
    """An agent controlled by a human player via standard console input.

    Attributes:
        name: Display name for this player, shown in prompts and reports.
    """

    def __init__(self, name: str = "Human") -> None:
        """Initialise the human agent.

        Args:
            name: A display name for the player, e.g. "Player 1".
        """
        self.name = name

    def choose_move(self, state: GameState) -> Move:
        """Prompt the human for a move and validate it.

        Displays the current board before each prompt. Loops until the
        user provides a syntactically valid (row, col) pair that targets
        an empty cell.

        Args:
            state: The current game state. The board is printed for context.

        Returns:
            A validated (row, col) move chosen by the human.
        """
        print(Board.render(state.board, state.n, state.last_move))
        print(f"\n{self.name} ({state.current_player.value}), enter your move.")

        while True:
            raw = input("  Row and column (e.g. 1 2): ").strip()
            parsed = self._parse_input(raw, state.n)

            if parsed is None:
                print(
                    f"  Invalid input. Enter two integers between 0 and {state.n - 1}."
                )
                continue

            row, col = parsed
            if not self.validate_move(state, (row, col)):
                print(f"  Cell ({row}, {col}) is occupied or out of range. Try again.")
                continue

            break

        # Zero out instrumentation — humans do not search.
        state.nodes_visited = 0
        state.max_depth_reached = 0
        state.prunings = 0
        state.compute_ebf()

        return (row, col)

    def get_name(self) -> str:
        """Return the display name of this human player.

        Returns:
            The name string provided at construction.
        """
        return self.name

    def get_tier(self) -> int:
        """Return the baseline tier for human players.

        Returns:
            0, indicating a non-algorithmic baseline.
        """
        return 0

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _parse_input(raw: str, n: int) -> tuple[int, int] | None:
        """Attempt to parse a raw string into a (row, col) pair.

        Args:
            raw: The user's raw input string.
            n: Board dimension; used to range-check the parsed values.

        Returns:
            A (row, col) tuple if parsing succeeds and values are in range,
            otherwise None.
        """
        parts = raw.split()
        if len(parts) != 2:
            return None

        try:
            row = int(parts[0])
            col = int(parts[1])
        except ValueError:
            return None

        if not (0 <= row < n and 0 <= col < n):
            return None

        return (row, col)
