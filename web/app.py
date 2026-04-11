"""TicTacToe Web UI — FastAPI application.

Run from the project root:
    uvicorn web.app:app --reload --port 8000

Then open http://localhost:8000 in a browser.
"""
from __future__ import annotations

import asyncio
import pathlib
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tictactoe.config import ConfigError, load_config
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Player

from web.agent_registry import AGENTS, create_agent, make_match_config
from web.database import get_history, get_scores, init_db, save_game
from web.game_session import (
    apply_agent_move,
    apply_human_move,
    create_session,
    get_session,
    remove_session,
    session_to_dict,
)
from web.probability import compute_win_probability

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_STATIC_DIR = pathlib.Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = _PROJECT_ROOT / "config.toml"
    try:
        load_config(config_path)
    except ConfigError as exc:
        # Agents that read from config will fall back to their own defaults.
        print(f"[web] Warning: config.toml issue — {exc}")
    init_db()
    yield


app = FastAPI(title="TicTacToe AI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class NewGameRequest(BaseModel):
    mode: str = "hva"          # 'hvh' | 'hva' | 'ava'
    player_side: str = "X"     # 'X' | 'O'  (hva only)
    agent_x: str = "mcts_vanilla"
    agent_o: str = "mcts_vanilla"
    n: int = Field(default=3, ge=3, le=15)
    k: int = Field(default=0, ge=0, le=15)   # 0 = use n
    match_mode: str = "time"                  # 'time' | 'node' | 'depth'
    time_limit_ms: float = Field(default=1000.0, gt=0)
    node_budget: int = Field(default=100_000, ge=1)
    fixed_depth: int = Field(default=4, ge=1, le=20)


class MoveRequest(BaseModel):
    row: int
    col: int


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_response(game_id: str) -> dict[str, Any]:
    session = get_session(game_id)
    prob = compute_win_probability(session.state)
    return session_to_dict(session, prob)


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------

@app.get("/api/agents")
async def list_agents():
    return AGENTS


# ---------------------------------------------------------------------------
# Config endpoint
# ---------------------------------------------------------------------------

@app.get("/api/config")
async def get_config_values():
    try:
        from tictactoe.config import get_config
        cfg = get_config()
        return {
            "search": {
                "id_max_depth": cfg.search.id_max_depth,
                "aspiration_delta": cfg.search.aspiration_delta,
                "tt_size": cfg.search.tt_size,
                "tss_max_depth": cfg.search.tss_max_depth,
                "mtdf_max_iterations": cfg.search.mtdf_max_iterations,
            },
            "budget": {
                "time_limit_ms": cfg.budget.time_limit_ms,
                "node_budget": cfg.budget.node_budget,
                "fixed_depth": cfg.budget.fixed_depth,
            },
            "mcts": {
                "exploration_constant": cfg.mcts.exploration_constant,
                "rollout_depth_limit": cfg.mcts.rollout_depth_limit,
                "rave_k": cfg.mcts.rave_k,
                "rave_exploration_constant": cfg.mcts.rave_exploration_constant,
                "heuristic_rollout_depth": cfg.mcts.heuristic_rollout_depth,
                "alphazero_lite_simulations": cfg.mcts.alphazero_lite_simulations,
                "alphazero_lite_c_puct": cfg.mcts.alphazero_lite_c_puct,
            },
            "game": {
                "n": cfg.game.n,
                "k": cfg.game.k,
                "num_games": cfg.game.num_games,
                "seed": cfg.game.seed,
            },
        }
    except ConfigError:
        return {}


# ---------------------------------------------------------------------------
# Game endpoints
# ---------------------------------------------------------------------------

@app.post("/api/game/new")
async def new_game(req: NewGameRequest):
    k = req.k if req.k > 0 else req.n
    k = min(k, req.n)

    match_config = make_match_config(
        req.match_mode, req.time_limit_ms, req.node_budget, req.fixed_depth
    )

    # Build agents based on mode
    if req.mode == "hvh":
        agent_x = agent_o = None
        player_x_name = "Human (X)"
        player_o_name = "Human (O)"

    elif req.mode == "hva":
        if req.player_side.upper() == "X":
            agent_x = None
            agent_o = create_agent(req.agent_o, match_config)
            player_x_name = "Human"
            agent_x_meta = next((a for a in AGENTS if a["id"] == req.agent_o), None)
            player_o_name = agent_x_meta["name"] if agent_x_meta else req.agent_o
        else:
            agent_x = create_agent(req.agent_x, match_config)
            agent_o = None
            agent_x_meta = next((a for a in AGENTS if a["id"] == req.agent_x), None)
            player_x_name = agent_x_meta["name"] if agent_x_meta else req.agent_x
            player_o_name = "Human"

    elif req.mode == "ava":
        agent_x = create_agent(req.agent_x, match_config)
        agent_o = create_agent(req.agent_o, match_config)
        ax_meta = next((a for a in AGENTS if a["id"] == req.agent_x), None)
        ao_meta = next((a for a in AGENTS if a["id"] == req.agent_o), None)
        player_x_name = ax_meta["name"] if ax_meta else req.agent_x
        player_o_name = ao_meta["name"] if ao_meta else req.agent_o

    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode!r}")

    state = GameState(
        board=Board.create(req.n),
        current_player=Player.X,
        n=req.n,
        k=k,
    )
    session = create_session(
        mode=req.mode,
        state=state,
        player_x_name=player_x_name,
        player_o_name=player_o_name,
        agent_x=agent_x,
        agent_o=agent_o,
    )
    return _make_response(session.game_id)


@app.get("/api/game/{game_id}")
async def get_game(game_id: str):
    try:
        return _make_response(game_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Game not found")


@app.post("/api/game/{game_id}/move")
async def human_move(game_id: str, req: MoveRequest):
    try:
        session = get_session(game_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Game not found")
    try:
        apply_human_move(session, (req.row, req.col))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    resp = _make_response(game_id)
    if resp["game_over"]:
        _persist_and_remove(session)
    return resp


@app.post("/api/game/{game_id}/ai-step")
async def ai_step(game_id: str):
    """Run one AI move.  Used for both HvA (opponent reply) and AvA loops."""
    try:
        session = get_session(game_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Game not found")
    try:
        # Run synchronous agent in a thread so we don't block the event loop.
        await asyncio.to_thread(apply_agent_move, session)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    resp = _make_response(game_id)
    if resp["game_over"]:
        _persist_and_remove(session)
    return resp


@app.delete("/api/game/{game_id}")
async def abandon_game(game_id: str):
    """Abandon (resign) the current game without saving to history."""
    try:
        session = get_session(game_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Game not found")
    remove_session(game_id)
    return {"abandoned": True}


# ---------------------------------------------------------------------------
# History and scores
# ---------------------------------------------------------------------------

@app.get("/api/history")
async def history(limit: int = 50, offset: int = 0):
    return get_history(limit=limit, offset=offset)


@app.get("/api/scores")
async def scores():
    return get_scores()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _persist_and_remove(session) -> None:
    """Save a finished session to the DB then drop it from memory."""
    state = session.state
    save_game(
        mode=session.mode,
        player_x=session.player_x_name,
        player_o=session.player_o_name,
        result=state.result.name,
        moves=list(state.move_history),
        n=state.n,
        k=state.k,
        duration_ms=round((session.ended_at - session.started_at) * 1000)
        if session.ended_at else None,
    )
    remove_session(session.game_id)
