"""Game session manager for n×n Tic-Tac-Toe.

The Game class orchestrates a single match between two agents. It owns the
GameState, delegates move selection to agents, enforces timing, and detects
game termination via Board utilities.

Dependency chain position: types → state → board → game → agents → benchmark.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import (
    DEFAULT_TIME_LIMIT_MS,
    MatchMode,
    Player,
    Result,
)

if TYPE_CHECKING:
    from tictactoe.agents.base_agent import BaseAgent
    from tictactoe.benchmark.metrics import MatchConfig


class Game:
    """Manages a single game session between two agents.

    Attributes:
        agent_x: The agent playing as Player X.
        agent_o: The agent playing as Player O.
        n: Board dimension.
        k: Winning line length.
        match_config: Configuration controlling time/node/depth budgets.
        state: The current game state.
    """

    def __init__(
        self,
        agent_x: BaseAgent,
        agent_o: BaseAgent,
        n: int = 3,
        k: int | None = None,
        match_config: MatchConfig | None = None,
    ) -> None:
        """Initialise a game between two agents.

        Args:
            agent_x: Agent playing as Player X (moves first).
            agent_o: Agent playing as Player O.
            n: Board dimension (n×n). Defaults to 3.
            k: Winning line length. Defaults to n when None.
            match_config: Budget configuration for the match. When None,
                defaults to TIME_CONTROLLED with DEFAULT_TIME_LIMIT_MS.
        """
        self.agent_x = agent_x
        self.agent_o = agent_o
        self.n = n
        self.k = k if k is not None else n
        self.match_config = match_config or _default_match_config()
        self.state = self._fresh_state()

    # -------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the game to a clean initial state.

        The same agents and configuration are retained; only the board and
        all tracking fields are cleared.
        """
        self.state = self._fresh_state()

    def step(self) -> GameState:
        """Execute one move for the current player's agent.

        The agent's choose_move is timed. After the call:
        - state.time_taken_ms is set to the elapsed wall-clock time.
        - state.compute_ebf() is called to finalise the EBF metric.
        - state.time_limit_exceeded is set if the agent ran over budget
          (TIME_CONTROLLED mode only). The move is accepted regardless.
        - state.result is updated via Board.is_terminal.

        Returns:
            The updated GameState after the move has been applied.
        """
        agent = self._current_agent()

        start_ns = time.perf_counter_ns()
        chosen_move = agent.choose_move(self.state)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        # Capture instrumentation written by the agent BEFORE apply_move
        # resets them (apply_move returns a fresh state with all metrics at 0).
        nodes_visited = self.state.nodes_visited
        max_depth_reached = self.state.max_depth_reached
        prunings = self.state.prunings

        # Apply the move — returns a new state without mutating old one.
        self.state = self.state.apply_move(chosen_move)

        # Restore instrumentation to the new state and finalise metrics.
        self.state.nodes_visited = nodes_visited
        self.state.max_depth_reached = max_depth_reached
        self.state.prunings = prunings
        self.state.time_taken_ms = elapsed_ms
        self.state.compute_ebf()

        # Flag time budget violations in TIME_CONTROLLED mode.
        if (
            self.match_config.mode is MatchMode.TIME_CONTROLLED
            and elapsed_ms > self.match_config.time_limit_ms
        ):
            self.state.time_limit_exceeded = True

        # Determine whether the game has ended.
        self.state.result = Board.is_terminal(
            self.state.board, self.n, self.k, chosen_move
        )

        return self.state

    def run(self, verbose: bool = False) -> Result:
        """Play the game to completion.

        Args:
            verbose: When True, the board is printed after every move.

        Returns:
            The final Result (X_WINS, O_WINS, or DRAW).
        """
        while self.state.result is Result.IN_PROGRESS:
            self.step()
            if verbose:
                print(Board.render(self.state.board, self.n, self.state.last_move))
                print()

        return self.state.result

    def get_state(self) -> GameState:
        """Return a read-only snapshot of the current game state.

        Returns:
            A deep copy of the internal state so callers cannot mutate it.
        """
        return self.state.copy()

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _fresh_state(self) -> GameState:
        """Build a blank GameState for the start of a new game.

        Returns:
            A GameState with an empty n×n board and Player X to move first.
        """
        return GameState(
            board=Board.create(self.n),
            current_player=Player.X,
            n=self.n,
            k=self.k,
        )

    def _current_agent(self) -> BaseAgent:
        """Return the agent whose turn it is.

        Returns:
            agent_x when it is Player X's turn, agent_o otherwise.
        """
        if self.state.current_player is Player.X:
            return self.agent_x
        return self.agent_o


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _default_match_config() -> MatchConfig:
    """Create the default MatchConfig used when none is supplied to Game.

    Deferred import avoids a circular dependency between game.py and metrics.py
    while keeping the public interface clean.

    Returns:
        A TIME_CONTROLLED MatchConfig with DEFAULT_TIME_LIMIT_MS per move.
    """
    from tictactoe.benchmark.metrics import MatchConfig  # noqa: PLC0415

    return MatchConfig.time_controlled(DEFAULT_TIME_LIMIT_MS)
