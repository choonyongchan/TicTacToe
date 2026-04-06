"""Abstract base class that every agent implementation must inherit from.

All agents — human, random, minimax, MCTS, etc. — share this interface.
The benchmarking layer depends only on this contract, not on any concrete
agent class.

Dependency chain position: types → state → board → game → agents → benchmark.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Move


class BaseAgent(ABC):
    """Contract that every agent in the framework must satisfy.

    Subclasses implement game-specific logic in choose_move and provide
    metadata via get_name and get_tier. The base class supplies shared
    utilities (move validation, instrumentation export) that agents should
    use rather than re-implement.
    """

    # -------------------------------------------------------------------
    # Abstract interface — every subclass must implement these
    # -------------------------------------------------------------------

    @abstractmethod
    def choose_move(self, state: GameState) -> Move:
        """Select a move given the current game state.

        Before returning, implementations MUST:
        - Set state.nodes_visited to the number of search nodes expanded.
        - Set state.max_depth_reached to the deepest ply searched.
        - Set state.prunings to the number of pruning events that occurred.
        - Call state.compute_ebf() to finalise the effective branching factor.

        Args:
            state: The current game state. Agents may read but must not
                mutate this object.

        Returns:
            A (row, col) tuple representing the chosen move. The move must
            be a legal empty cell on the current board.
        """

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable identifier for this agent.

        Used in benchmark reports and Elo rankings to label results.

        Returns:
            A descriptive name string, e.g. "Minimax-AB depth=5".
        """

    @abstractmethod
    def get_tier(self) -> int:
        """Return the algorithm tier this agent belongs to.

        Tiers are used by the reporter to group and compare algorithms:
        - 0: Baselines (RandomAgent, HumanAgent, BruteForceOracle)
        - 1: Exact search (Minimax, NegaMax, NegaScout, MTD(f))
        - 2: Enhanced search with heuristics
        - 3: Probabilistic / approximate search (MCTS, etc.)
        - 4: Learning-based approaches

        Returns:
            An integer tier label.
        """

    # -------------------------------------------------------------------
    # Concrete utilities — agents may override but typically should not
    # -------------------------------------------------------------------

    def validate_move(self, state: GameState, move: Move) -> bool:
        """Check whether a move is legal on the current board.

        A move is legal when the target cell is within bounds and empty.

        Args:
            state: The current game state.
            move: The (row, col) pair to validate.

        Returns:
            True if the move is a legal empty cell, False otherwise.
        """
        row, col = move
        in_bounds: bool = 0 <= row < state.n and 0 <= col < state.n
        is_empty: bool = state.board[row][col] is Cell.EMPTY
        return in_bounds and is_empty

    def get_instrumentation(self) -> dict[str, float | int]:
        """Return the instrumentation fields from the last choose_move call.

        This method is a convenience wrapper for the benchmarking layer. It
        should be called after choose_move has returned and state fields
        have been populated.

        Note:
            Because instrumentation lives on the GameState object passed to
            choose_move, this method returns an empty dict by default. Agents
            that cache their last state can override this for direct access.

        Returns:
            A dictionary with keys: nodes_visited, max_depth_reached,
            prunings, time_taken_ms, effective_branching_factor.
            Returns an empty dict in the base implementation.
        """
        return {}
