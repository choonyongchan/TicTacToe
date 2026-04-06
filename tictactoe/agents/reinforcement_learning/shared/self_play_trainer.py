"""Self-play trainer utility for RL agents.

Provides a simple self-play loop that generates game data by having an agent
play against itself, collecting (state, policy, value) training examples.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player, Result

if TYPE_CHECKING:
    from tictactoe.agents.base_agent import BaseAgent


class SelfPlayTrainer:
    """Generates self-play game data for training RL agents.

    Each episode produces a sequence of (state_tensor, policy, outcome)
    tuples suitable for supervised training of a PolicyValueNetwork.
    """

    def __init__(self, n: int = 3, k: int | None = None, seed: int | None = None) -> None:
        self.n = n
        self.k = k if k is not None else n
        self._rng = random.Random(seed)

    def play_episode(
        self,
        agent_x: "BaseAgent",
        agent_o: "BaseAgent",
    ) -> tuple[list, Result]:
        """Play one complete game and return move history plus result.

        Args:
            agent_x: Agent playing as X.
            agent_o: Agent playing as O.

        Returns:
            (move_history, result) where move_history is the list of moves
            and result is the final game outcome.
        """
        board = Board.create(self.n)
        state = GameState(
            board=board,
            current_player=Player.X,
            n=self.n,
            k=self.k,
        )
        agents = {Player.X: agent_x, Player.O: agent_o}
        move_history = []

        for _ in range(self.n * self.n):
            if state.result != Result.IN_PROGRESS:
                break
            agent = agents[state.current_player]
            move = agent.choose_move(state)
            state = state.apply_move(move)
            state.result = Board.is_terminal(
                state.board, state.n, state.k, state.last_move
            )
            move_history.append(move)

        return move_history, state.result
