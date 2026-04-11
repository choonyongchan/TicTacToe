"""In-memory game session management.

Each active game is stored as a GameSession.  The module exposes helpers to
create sessions, apply human moves, and trigger agent moves.  Finished games
are not kept in memory — callers are responsible for persisting them via
web.database before dropping the reference.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Move, Player, Result


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class GameSession:
    game_id: str
    mode: str           # 'hvh' | 'hva' | 'ava'
    state: GameState
    player_x_name: str  # display name: 'Human' or agent label
    player_o_name: str
    agent_x: BaseAgent | None   # None when player is human
    agent_o: BaseAgent | None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None


# ---------------------------------------------------------------------------
# Active session store
# ---------------------------------------------------------------------------

_sessions: dict[str, GameSession] = {}


def get_session(game_id: str) -> GameSession:
    session = _sessions.get(game_id)
    if session is None:
        raise KeyError(f"No active session: {game_id!r}")
    return session


def create_session(
    mode: str,
    state: GameState,
    player_x_name: str,
    player_o_name: str,
    agent_x: BaseAgent | None,
    agent_o: BaseAgent | None,
) -> GameSession:
    game_id = str(uuid.uuid4())
    session = GameSession(
        game_id=game_id,
        mode=mode,
        state=state,
        player_x_name=player_x_name,
        player_o_name=player_o_name,
        agent_x=agent_x,
        agent_o=agent_o,
    )
    _sessions[game_id] = session
    return session


def remove_session(game_id: str) -> None:
    _sessions.pop(game_id, None)


# ---------------------------------------------------------------------------
# Move application helpers
# ---------------------------------------------------------------------------

def _apply_and_check(state: GameState, move: Move) -> GameState:
    """Apply move to state, detect terminal result, return new state."""
    new_state = state.apply_move(move)
    result = Board.is_terminal(new_state.board, new_state.n, new_state.k, move)
    if result is not Result.IN_PROGRESS:
        new_state.result = result
    return new_state


def apply_human_move(session: GameSession, move: Move) -> GameState:
    """Apply a human player's move.  Raises ValueError on illegal input."""
    state = session.state
    row, col = move

    if state.result is not Result.IN_PROGRESS:
        raise ValueError("Game is already over.")

    # Determine whose turn it is
    current = state.current_player
    if session.mode == "hva":
        human_side = Player.X if session.agent_x is None else Player.O
        if current is not human_side:
            raise ValueError("It is not the human player's turn.")
    elif session.mode == "ava":
        raise ValueError("No human moves in AI-vs-AI mode.")

    if not (0 <= row < state.n and 0 <= col < state.n):
        raise ValueError(f"Move ({row},{col}) is out of bounds for {state.n}×{state.n} board.")
    if not Board.is_empty(state.board, row, col):
        raise ValueError(f"Cell ({row},{col}) is already occupied.")

    session.state = _apply_and_check(state, move)
    if session.state.result is not Result.IN_PROGRESS:
        session.ended_at = time.time()
    return session.state


def apply_agent_move(session: GameSession) -> GameState:
    """Run the current player's agent and apply its move synchronously.

    Raises ValueError if the current player is human or the game is over.
    """
    state = session.state

    if state.result is not Result.IN_PROGRESS:
        raise ValueError("Game is already over.")

    current = state.current_player
    agent = session.agent_x if current is Player.X else session.agent_o
    if agent is None:
        raise ValueError(f"Player {current.name} is human — use apply_human_move.")

    move = agent.choose_move(state)
    session.state = _apply_and_check(state, move)
    if session.state.result is not Result.IN_PROGRESS:
        session.ended_at = time.time()
    return session.state


def session_to_dict(session: GameSession, probability_x: float) -> dict[str, Any]:
    """Serialise a session to a JSON-compatible dict for API responses."""
    state = session.state
    return {
        "game_id": session.game_id,
        "mode": session.mode,
        "player_x_name": session.player_x_name,
        "player_o_name": session.player_o_name,
        "state": state.to_dict(),
        "probability_x": round(probability_x, 4),
        "game_over": state.result is not Result.IN_PROGRESS,
        "result": state.result.name,
        "duration_ms": (
            round((session.ended_at - session.started_at) * 1000)
            if session.ended_at else None
        ),
    }
