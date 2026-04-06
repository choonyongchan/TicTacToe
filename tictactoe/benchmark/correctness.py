"""Ground-truth correctness verification for n=3 Tic-Tac-Toe.

BruteForceOracle solves 3×3 Tic-Tac-Toe exhaustively via plain minimax with
no alpha-beta pruning and no heuristics. Its correctness is therefore
straightforward to audit.

The internal search works directly on a flat board array with explicit
make/unmake moves to avoid the overhead of deep-copying GameState on every
recursive call. This keeps the implementation fast while remaining
transparent and easy to verify.

KNOWN_POSITIONS provides a curated library of board positions with provably
correct expected outcomes, used for move-level correctness testing of any agent.
"""

from __future__ import annotations

import logging
from typing import Optional

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import (
    Board2D,
    Cell,
    Move,
    Player,
    Result,
)

logger = logging.getLogger(__name__)

# Internal cell constants used by the fast board representation.
_EMPTY = 0
_X = 1
_O = 2

_WIN_LINES_3x3: list[tuple[int, int, int]] = [
    # Rows
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    # Columns
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    # Diagonals
    (0, 4, 8), (2, 4, 6),
]


# ---------------------------------------------------------------------------
# BruteForceOracle
# ---------------------------------------------------------------------------


class BruteForceOracle(BaseAgent):
    """Exhaustive minimax solver for 3×3 Tic-Tac-Toe.

    This agent is NOT a general algorithm — it is hardcoded for n=3 and
    has no depth limit. It exists solely as a correctness reference: on the
    solved 3×3 game, two optimal players always draw.

    Uses plain minimax (no alpha-beta, no heuristics) so that its logic is
    transparent and beyond reasonable doubt.

    The internal search operates on a flat integer array with explicit
    make/unmake moves rather than deep-copying GameState on each node,
    keeping it fast enough for the test suite without sacrificing clarity.
    """

    def choose_move(self, state: GameState) -> Move:
        """Return the provably optimal move for the current player.

        Performs an exhaustive minimax search over all legal continuations.
        On n=3, optimal play always results in a draw.

        Args:
            state: The current game state. Must be a 3×3 board.

        Returns:
            The (row, col) of the optimal move.

        Raises:
            ValueError: If the board is not 3×3 or has no legal moves.
        """
        if state.n != 3:
            raise ValueError(
                f"BruteForceOracle only supports n=3, got n={state.n}."
            )

        # Convert the Board2D to a flat integer array for fast access.
        flat_board = _board2d_to_flat(state.board)
        root_player = _player_to_int(state.current_player)

        nodes_visited = [0]  # Mutable counter passed into recursion.
        best_move, _ = _minimax(flat_board, root_player, root_player, nodes_visited)

        state.nodes_visited = nodes_visited[0]
        state.max_depth_reached = 9 - flat_board.count(_EMPTY) + len(
            [c for c in flat_board if c == _EMPTY]
        )
        state.prunings = 0
        state.compute_ebf()

        if best_move is None:
            raise ValueError("BruteForceOracle: no legal moves available.")

        return (best_move // 3, best_move % 3)

    def get_name(self) -> str:
        """Return the display name for this oracle agent.

        Returns:
            The string "BruteForceOracle-n3".
        """
        return "BruteForceOracle-n3"

    def get_tier(self) -> int:
        """Return the baseline tier for the oracle.

        Returns:
            0, indicating a correctness reference, not a research algorithm.
        """
        return 0


# ---------------------------------------------------------------------------
# Fast minimax on flat board
# ---------------------------------------------------------------------------


def _minimax(
    board: list[int],
    current_player: int,
    root_player: int,
    nodes_visited: list[int],
) -> tuple[Optional[int], float]:
    """Recursive minimax search on a flat 3×3 integer board.

    Uses make/unmake moves instead of copying the board on each call.
    Scores are from root_player's perspective: +1 = win, -1 = loss, 0 = draw.

    Args:
        board: Flat 9-element list of _EMPTY/_X/_O values.
        current_player: _X or _O — whose turn it is.
        root_player: _X or _O — the player at the search root.
        nodes_visited: Single-element list used as a mutable counter.

    Returns:
        A (best_index, score) pair where best_index is the flat board index
        of the chosen move, or None at terminal nodes.
    """
    nodes_visited[0] += 1

    winner = _check_winner(board)
    if winner != _EMPTY:
        return None, 1.0 if winner == root_player else -1.0

    empty_cells = [i for i, c in enumerate(board) if c == _EMPTY]
    if not empty_cells:
        return None, 0.0

    opponent = _O if current_player == _X else _X
    is_maximising = current_player == root_player

    best_index: Optional[int] = None
    best_score = float("-inf") if is_maximising else float("inf")

    for idx in empty_cells:
        board[idx] = current_player
        _, child_score = _minimax(board, opponent, root_player, nodes_visited)
        board[idx] = _EMPTY  # Undo move.

        if is_maximising and child_score > best_score:
            best_score = child_score
            best_index = idx
        elif not is_maximising and child_score < best_score:
            best_score = child_score
            best_index = idx

    return best_index, best_score


def _check_winner(board: list[int]) -> int:
    """Check for a winner in a flat 3×3 board.

    Args:
        board: Flat 9-element integer board.

    Returns:
        _X, _O, or _EMPTY (meaning no winner yet).
    """
    for a, b, c in _WIN_LINES_3x3:
        if board[a] != _EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return _EMPTY


def _board2d_to_flat(board: Board2D) -> list[int]:
    """Convert a 3×3 Board2D to a flat integer list.

    Args:
        board: The Board2D to convert.

    Returns:
        A 9-element list of _EMPTY/_X/_O integers.
    """
    mapping = {Cell.EMPTY: _EMPTY, Cell.X: _X, Cell.O: _O}
    return [mapping[board[r][c]] for r in range(3) for c in range(3)]


def _player_to_int(player: Player) -> int:
    """Convert a Player enum to the internal integer representation.

    Args:
        player: Player.X or Player.O.

    Returns:
        _X or _O.
    """
    return _X if player is Player.X else _O


# ---------------------------------------------------------------------------
# Known positions library
# ---------------------------------------------------------------------------


def _board_from_ints(rows: list[list[int]]) -> Board2D:
    """Convert a list-of-ints board to a Board2D of Cell values.

    Args:
        rows: 3×3 list where 0=EMPTY, 1=X, 2=O.

    Returns:
        A Board2D using Cell enum values.
    """
    mapping = {0: Cell.EMPTY, 1: Cell.X, 2: Cell.O}
    return [[mapping[val] for val in row] for row in rows]


KNOWN_POSITIONS: list[dict] = [
    # ---------------------------------------------------------------
    # 1. Empty board — first move; any non-losing move is acceptable.
    #    X=0, O=0 → X's turn. ✓
    # ---------------------------------------------------------------
    {
        "description": "Empty board — first move",
        "board": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "current_player": "X",
        "expected_move": None,
        "expected_result": "DRAW",
        "must_not_lose": True,
    },
    # ---------------------------------------------------------------
    # 2. X wins immediately (horizontal row 0).
    #    X at (0,0),(0,1) → wins at (0,2). O at (1,1),(2,0).
    #    X=2, O=2 → X's turn (total=4, even). ✓
    # ---------------------------------------------------------------
    {
        "description": "X wins immediately at (0,2)",
        "board": [[1, 1, 0], [0, 2, 0], [2, 0, 0]],
        "current_player": "X",
        "expected_move": (0, 2),
        "expected_result": "X_WINS",
        "must_not_lose": False,
    },
    # ---------------------------------------------------------------
    # 3. O wins immediately (anti-diagonal).
    #    O at (0,2),(1,1) → wins at (2,0). X at (0,0),(0,1),(1,0).
    #    X=3, O=2 → O's turn (total=5, odd). ✓
    # ---------------------------------------------------------------
    {
        "description": "O wins immediately at (2,0)",
        "board": [[1, 1, 2], [1, 2, 0], [0, 0, 0]],
        "current_player": "O",
        "expected_move": (2, 0),
        "expected_result": "O_WINS",
        "must_not_lose": False,
    },
    # ---------------------------------------------------------------
    # 4. X must block O's immediate win (row 0).
    #    O at (0,0),(0,1) → threatens (0,2). X at (1,1),(2,2) — no immediate win.
    #    X=2, O=2 → X's turn (total=4, even). ✓
    # ---------------------------------------------------------------
    {
        "description": "X must block O at (0,2)",
        "board": [[2, 2, 0], [0, 1, 0], [0, 0, 1]],
        "current_player": "X",
        "expected_move": (0, 2),
        "expected_result": "DRAW",
        "must_not_lose": False,
    },
    # ---------------------------------------------------------------
    # 5. O must block X's immediate win (column 0).
    #    X at (0,0),(1,0),(2,1) → threatens only (2,0). O at (0,1),(2,2).
    #    X=3, O=2 → O's turn (total=5, odd). ✓
    #    All O moves except (2,0) allow X to win col 0 immediately.
    # ---------------------------------------------------------------
    {
        "description": "O must block X at (2,0)",
        "board": [[1, 2, 0], [1, 0, 0], [0, 1, 2]],
        "current_player": "O",
        "expected_move": (2, 0),
        "expected_result": "DRAW",
        "must_not_lose": False,
    },
    # ---------------------------------------------------------------
    # 6. Second move: O played a corner; X should play optimally.
    #    X=1, O=1 → X's turn (total=2, even). must_not_lose=True.
    # ---------------------------------------------------------------
    {
        "description": "X second move — O took a corner, play optimally",
        "board": [[2, 0, 0], [0, 0, 0], [0, 0, 1]],
        "current_player": "X",
        "expected_move": None,
        "expected_result": "DRAW",
        "must_not_lose": True,
    },
    # ---------------------------------------------------------------
    # 7. X wins immediately on main diagonal.
    #    X at (0,0),(2,2) → wins at (1,1) (diagonal). O at (0,2),(2,0).
    #    X=2, O=2 → X's turn (total=4, even). ✓
    # ---------------------------------------------------------------
    {
        "description": "X wins immediately on diagonal at (1,1)",
        "board": [[1, 0, 2], [0, 0, 0], [2, 0, 1]],
        "current_player": "X",
        "expected_move": (1, 1),
        "expected_result": "X_WINS",
        "must_not_lose": False,
    },
    # ---------------------------------------------------------------
    # 8. Fork defence: O must not allow X a fork.
    #    X at (0,0),(2,2), O at (1,1). X=2, O=1 → O's turn (total=3, odd). ✓
    # ---------------------------------------------------------------
    {
        "description": "O defends against X fork — must_not_lose",
        "board": [[1, 0, 0], [0, 2, 0], [0, 0, 1]],
        "current_player": "O",
        "expected_move": None,
        "expected_result": "DRAW",
        "must_not_lose": True,
    },
    # ---------------------------------------------------------------
    # 9. Near-draw endgame: only one legal move remains.
    #    X=4, O=4 → X's turn (total=8, even). X plays (2,2) → DRAW. ✓
    # ---------------------------------------------------------------
    {
        "description": "Near-draw: one empty cell at (2,2)",
        "board": [[1, 2, 1], [1, 2, 2], [2, 1, 0]],
        "current_player": "X",
        "expected_move": (2, 2),
        "expected_result": "DRAW",
        "must_not_lose": False,
    },
    # ---------------------------------------------------------------
    # 10. X wins immediately — multiple winning moves exist.
    #     X at (0,0),(1,1); O at (0,1),(1,0). X can fork via (0,2)
    #     (anti-diagonal + main diagonal threats), winning first.
    #     X=2, O=2 → X's turn (total=4, even). ✓
    # ---------------------------------------------------------------
    {
        "description": "X wins via fork at (0,2)",
        "board": [[1, 2, 0], [2, 1, 0], [0, 0, 0]],
        "current_player": "X",
        "expected_move": (0, 2),
        "expected_result": "X_WINS",
        "must_not_lose": False,
    },
    # ---------------------------------------------------------------
    # 11. X must block O's anti-diagonal threat.
    #     O at (0,2),(1,1) → threatens (2,0). X at (0,0),(0,1) — no immediate win.
    #     X=2, O=2 → X's turn (total=4, even). ✓
    # ---------------------------------------------------------------
    {
        "description": "X blocks O anti-diagonal at (2,0)",
        "board": [[1, 1, 2], [0, 2, 0], [0, 0, 0]],
        "current_player": "X",
        "expected_move": (2, 0),
        "expected_result": "DRAW",
        "must_not_lose": False,
    },
    # ---------------------------------------------------------------
    # 12. O wins immediately — multiple winning moves exist.
    #     O at (0,2),(1,2); X at (0,0),(0,1),(1,0). O can win via (2,0)
    #     which forces a fork (all X replies allow O to complete col 2).
    #     X=3, O=2 → O's turn (total=5, odd). ✓
    # ---------------------------------------------------------------
    {
        "description": "O wins via fork at (2,0)",
        "board": [[1, 1, 2], [1, 0, 2], [0, 0, 0]],
        "current_player": "O",
        "expected_move": (2, 0),
        "expected_result": "O_WINS",
        "must_not_lose": False,
    },
]


# ---------------------------------------------------------------------------
# Verification functions
# ---------------------------------------------------------------------------


def verify_agent_on_known_positions(agent: BaseAgent) -> dict:
    """Test an agent against the KNOWN_POSITIONS library.

    For each position the agent is asked to choose a move. The move is
    checked against the expected optimal move (when specified) and the
    expected outcome.

    Args:
        agent: The agent to evaluate.

    Returns:
        A dict with keys:
            "passed" (bool): True if all tests pass.
            "total" (int): Total positions tested.
            "correct" (int): Number of positions answered correctly.
            "failures" (list[dict]): Entries where the agent answered incorrectly.
    """
    total = len(KNOWN_POSITIONS)
    correct = 0
    failures: list[dict] = []

    for position in KNOWN_POSITIONS:
        board = _board_from_ints(position["board"])
        player = Player.X if position["current_player"] == "X" else Player.O

        state = GameState(
            board=board,
            current_player=player,
            n=3,
            k=3,
        )
        state.result = Board.is_terminal(board, 3, 3, None)

        try:
            chosen = agent.choose_move(state)
        except Exception as exc:
            failures.append({
                "description": position["description"],
                "error": str(exc),
            })
            continue

        if not _validate_known_position(position, state, chosen):
            failures.append({
                "description": position["description"],
                "expected_move": position["expected_move"],
                "chosen_move": chosen,
            })
        else:
            correct += 1

    passed = len(failures) == 0
    return {"passed": passed, "total": total, "correct": correct, "failures": failures}


def verify_oracle_never_loses(
    agent: BaseAgent, games: int = 100
) -> dict:
    """Play the agent against BruteForceOracle and check it never loses.

    On the solved 3×3 game, optimal play always draws. Any loss for the
    agent indicates a bug in the agent. Any "win" against the oracle
    indicates a bug in the oracle.

    Args:
        agent: The agent to test.
        games: Total number of games to play (split evenly across sides).

    Returns:
        A dict with keys:
            "passed" (bool): True if agent_losses == 0.
            "agent_losses" (int): Number of games where the agent lost.
            "draws" (int): Number of drawn games.
            "agent_wins" (int): Number of games where the agent won
                (should be 0; a win indicates an oracle bug).
            "games" (int): Total games played.
    """
    from tictactoe.core.game import Game

    oracle = BruteForceOracle()
    agent_losses = 0
    agent_wins = 0
    draws = 0

    for game_index in range(games):
        if game_index % 2 == 0:
            game = Game(agent_x=agent, agent_o=oracle, n=3, k=3)
            agent_is_x = True
        else:
            game = Game(agent_x=oracle, agent_o=agent, n=3, k=3)
            agent_is_x = False

        result = game.run()

        if result is Result.DRAW:
            draws += 1
        elif (result is Result.X_WINS and agent_is_x) or (
            result is Result.O_WINS and not agent_is_x
        ):
            agent_wins += 1
            logger.warning(
                "Agent %s beat BruteForceOracle — this likely indicates an "
                "oracle bug, not a super-human agent.",
                agent.get_name(),
            )
        else:
            agent_losses += 1

    passed = agent_losses == 0
    return {
        "passed": passed,
        "agent_losses": agent_losses,
        "draws": draws,
        "agent_wins": agent_wins,
        "games": games,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_known_position(
    position: dict, state: GameState, chosen_move: Move
) -> bool:
    """Check whether an agent's chosen move is acceptable for a known position.

    Args:
        position: The KNOWN_POSITIONS entry.
        state: The game state presented to the agent.
        chosen_move: The move selected by the agent.

    Returns:
        True if the move is acceptable (correct or within must_not_lose logic).
    """
    if position["must_not_lose"]:
        return _does_not_immediately_lose(state, chosen_move)

    expected = position["expected_move"]
    if expected is None:
        return Board.is_empty(state.board, chosen_move[0], chosen_move[1])

    return chosen_move == tuple(expected)


def _does_not_immediately_lose(state: GameState, move: Move) -> bool:
    """Check that playing a move does not immediately hand a win to the opponent.

    Args:
        state: The current game state.
        move: The move to evaluate.

    Returns:
        True if the move is legal and does not lose immediately.
    """
    if not Board.is_empty(state.board, move[0], move[1]):
        return False

    next_state = state.apply_move(move)
    next_state.result = Board.is_terminal(
        next_state.board, next_state.n, next_state.k, move
    )

    opponent = state.current_player.opponent()
    win_move = Board.get_winning_move(
        next_state.board, next_state.n, next_state.k, opponent
    )
    return win_move is None
