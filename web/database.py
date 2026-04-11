"""Persistent game history — separate feature module.

Stores completed games and aggregated win counts in a local SQLite database.
The file is created automatically at web/data/games.db on first start.

This module has no dependency on tictactoe.* and can be imported independently.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

_DB_PATH = pathlib.Path(__file__).parent / "data" / "games.db"
_lock = threading.Lock()


@contextmanager
def _db():
    with _lock:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db() -> None:
    """Create tables if they do not already exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT    NOT NULL,
                mode        TEXT    NOT NULL,
                player_x    TEXT    NOT NULL,
                player_o    TEXT    NOT NULL,
                result      TEXT,
                moves       TEXT    NOT NULL DEFAULT '[]',
                n           INTEGER NOT NULL,
                k           INTEGER NOT NULL,
                duration_ms INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                player_name TEXT    PRIMARY KEY,
                wins        INTEGER NOT NULL DEFAULT 0,
                losses      INTEGER NOT NULL DEFAULT 0,
                draws       INTEGER NOT NULL DEFAULT 0
            )
        """)


def save_game(
    *,
    mode: str,
    player_x: str,
    player_o: str,
    result: str,
    moves: list,
    n: int,
    k: int,
    duration_ms: int | None = None,
) -> int:
    """Persist a finished game and update per-player win/loss/draw tallies.

    Returns the new row id.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    with _db() as conn:
        cur = conn.execute(
            """INSERT INTO games
               (created_at, mode, player_x, player_o, result, moves, n, k, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (created_at, mode, player_x, player_o, result,
             json.dumps(moves), n, k, duration_ms),
        )
        game_id = cur.lastrowid

        def _upsert(player: str, won: bool, drew: bool) -> None:
            conn.execute(
                """INSERT INTO scores (player_name) VALUES (?)
                   ON CONFLICT(player_name) DO NOTHING""",
                (player,),
            )
            if drew:
                conn.execute(
                    "UPDATE scores SET draws = draws + 1 WHERE player_name = ?",
                    (player,),
                )
            elif won:
                conn.execute(
                    "UPDATE scores SET wins = wins + 1 WHERE player_name = ?",
                    (player,),
                )
            else:
                conn.execute(
                    "UPDATE scores SET losses = losses + 1 WHERE player_name = ?",
                    (player,),
                )

        if result == "DRAW":
            _upsert(player_x, False, True)
            _upsert(player_o, False, True)
        elif result == "X_WINS":
            _upsert(player_x, True, False)
            _upsert(player_o, False, False)
        elif result == "O_WINS":
            _upsert(player_x, False, False)
            _upsert(player_o, True, False)

    return game_id


def get_history(limit: int = 50, offset: int = 0) -> list[dict]:
    """Return recent games, newest first."""
    with _db() as conn:
        rows = conn.execute(
            """SELECT id, created_at, mode, player_x, player_o, result,
                      n, k, duration_ms
               FROM games ORDER BY id DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def get_scores() -> list[dict]:
    """Return all player score rows sorted by wins descending."""
    with _db() as conn:
        rows = conn.execute(
            """SELECT player_name, wins, losses, draws,
                      wins + losses + draws AS total_games
               FROM scores ORDER BY wins DESC, player_name ASC"""
        ).fetchall()
    return [dict(r) for r in rows]
