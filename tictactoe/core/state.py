"""GameState — the central, extensible state container for the framework.

Every agent receives and returns GameState instances. Subclasses may add
fields (e.g., Zobrist hashes, MCTS visit counts) without modifying existing
code. The only dependency is on core.types.

Dependency chain position: types → state → board → game → agents → benchmark.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

from tictactoe.core.types import Board2D, Cell, Move, Player, Result


@dataclass
class GameState:
    """The complete, self-contained description of a game at a single moment.

    Core fields describe the board position and move history. Instrumentation
    fields are populated by agents and used by the benchmarking layer.

    Attributes:
        board: Current board contents as a 2-D grid of Cell values.
        current_player: The player whose turn it is to move.
        n: The board dimension (n×n cells).
        k: Number of consecutive cells required to win.
        move_history: Ordered list of all moves made so far.
        last_move: The most recent move, or None at the start of the game.
        result: Current game outcome (IN_PROGRESS until the game ends).
        move_number: Total number of moves made so far.
        nodes_visited: Search nodes expanded during the last decision.
        max_depth_reached: Deepest ply searched during the last decision.
        time_taken_ms: Wall-clock milliseconds spent on the last decision.
        prunings: Alpha-beta or equivalent pruning events during last decision.
        effective_branching_factor: nodes_visited^(1/max_depth_reached),
            or 0.0 when max_depth_reached is zero.
        time_limit_exceeded: Set to True by Game.step when an agent exceeds
            its per-move time budget.
    """

    # --- Core fields ---
    board: Board2D
    current_player: Player
    n: int
    k: int
    move_history: list[Move] = field(default_factory=list)
    last_move: Move | None = None
    result: Result = Result.IN_PROGRESS
    move_number: int = 0

    # --- Instrumentation fields (agents must populate these) ---
    nodes_visited: int = 0
    max_depth_reached: int = 0
    time_taken_ms: float = 0.0
    prunings: int = 0
    effective_branching_factor: float = 0.0

    # --- Benchmark bookkeeping (set by Game.step, not agents) ---
    time_limit_exceeded: bool = False

    # ---------------------------------------------------------------
    # Public methods
    # ---------------------------------------------------------------

    def copy(self) -> GameState:
        """Return a deep copy of this state.

        The copy preserves all fields, including any attributes added by
        subclasses. No references are shared between the original and the copy.

        Returns:
            A new GameState (or subclass instance) identical to self.
        """
        return copy.deepcopy(self)

    def apply_move(self, move: Move) -> GameState:
        """Return a new state with *move* applied, without mutating self.

        The new state reflects:
        - The chosen cell marked with the current player's symbol.
        - move_history extended by move.
        - last_move set to move.
        - move_number incremented by one.
        - current_player switched to the opponent.

        Win detection is intentionally excluded — that is the responsibility
        of Board.is_terminal and Game.step.

        Args:
            move: A (row, col) pair identifying the cell to claim.

        Returns:
            A new GameState with the move applied.
        """
        new_state = self.copy()
        row, col = move

        # Update board — copy already made, so mutation is safe here.
        new_state.board[row][col] = self.current_player.to_cell()

        # Update tracking fields.
        new_state.move_history = self.move_history + [move]
        new_state.last_move = move
        new_state.move_number = self.move_number + 1
        new_state.current_player = self.current_player.opponent()

        # Reset instrumentation for the next decision.
        new_state.nodes_visited = 0
        new_state.max_depth_reached = 0
        new_state.time_taken_ms = 0.0
        new_state.prunings = 0
        new_state.effective_branching_factor = 0.0
        new_state.time_limit_exceeded = False

        return new_state

    def compute_ebf(self) -> float:
        """Compute and store the effective branching factor.

        EBF = nodes_visited ^ (1 / max_depth_reached).
        Returns and stores 0.0 when max_depth_reached is zero (e.g., random
        agents that do not perform any search).

        Returns:
            The computed effective branching factor.
        """
        if self.max_depth_reached == 0:
            self.effective_branching_factor = 0.0
        else:
            self.effective_branching_factor = math.pow(
                self.nodes_visited, 1.0 / self.max_depth_reached
            )
        return self.effective_branching_factor

    def to_dict(self) -> dict[str, Any]:
        """Serialise all fields to a JSON-compatible dictionary.

        The board is encoded as a 2-D list of cell names (strings). Enum
        values are converted to their string representations. All other
        values are left as-is (they are already JSON-serialisable primitives).

        Returns:
            A dictionary containing every field of the state.
        """
        return {
            "board": [
                [cell.name for cell in row] for row in self.board
            ],
            "current_player": self.current_player.name,
            "n": self.n,
            "k": self.k,
            "move_history": list(self.move_history),
            "last_move": self.last_move,
            "result": self.result.name,
            "move_number": self.move_number,
            "nodes_visited": self.nodes_visited,
            "max_depth_reached": self.max_depth_reached,
            "time_taken_ms": self.time_taken_ms,
            "prunings": self.prunings,
            "effective_branching_factor": self.effective_branching_factor,
            "time_limit_exceeded": self.time_limit_exceeded,
        }
