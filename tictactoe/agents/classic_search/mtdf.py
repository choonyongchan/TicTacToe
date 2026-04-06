"""MTD(f) — Tier 1 classic search with memory.

MTD(f) (Memory-enhanced Test Driver with value f) is an alpha-beta variant
that replaces iterative widening with iterative null-window (zero-width)
alpha-beta probes, each probe tightening the bounds [lower, upper] around
the true minimax value until they converge.

Algorithm:
    1. Start with an initial guess f (e.g. the heuristic evaluation of the
       root position).
    2. Call AlphaBetaWithMemory(state, depth, f-1, f) — a null-window probe
       centred at f.
    3. If the result < f, the true value <= result, so upper = result.
       Otherwise lower = result (a lower bound on the true value).
    4. Update f and repeat until lower >= upper (bounds have converged).

The transposition table (TT) is the key enabler: because each probe stores
and reuses results from prior probes, the total work is only slightly more
than a single full-window alpha-beta call of the same depth.

Key properties:
- Warm-starting: the first guess f can be taken from a previous iteration
  (e.g. iterative deepening) to reduce the number of probes needed.
- Per-move TT: the table is cleared at the start of each root move to avoid
  stale data from different board positions. (Compare with MTD(f)Enhanced,
  which uses a persistent TT shared across IDDFS iterations.)
- Zobrist hashing with a module-level random table supports boards up to 20×20.

Dependency chain: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import random
import time
from enum import Enum

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents._search_budget import SearchBudget
from tictactoe.agents._shared_utils import check_forced_move
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import INF, Cell, Move, Player, Result
from tictactoe.evaluation.heuristics import evaluate_position
from tictactoe.evaluation.move_ordering import order_moves
from tictactoe.benchmark.metrics import MatchConfig

# ---------------------------------------------------------------------------
# Zobrist table — seeded with Random(42), supports up to 20x20 boards
# ---------------------------------------------------------------------------

_ZOBRIST_RNG = random.Random(42)
_MAX_N = 20
# [row][col][player_index]: player_index 0=X, 1=O
_ZOBRIST_TABLE: list[list[list[int]]] = [
    [
        [_ZOBRIST_RNG.getrandbits(64), _ZOBRIST_RNG.getrandbits(64)]
        for _ in range(_MAX_N)
    ]
    for _ in range(_MAX_N)
]


def _hash_board(board: list, n: int) -> int:
    """Compute Zobrist hash of the current board.

    Args:
        board: 2D board of Cell values.
        n: Board dimension.

    Returns:
        64-bit integer hash.
    """
    h = 0
    for r in range(n):
        for c in range(n):
            cell = board[r][c]
            if cell is Cell.X:
                h ^= _ZOBRIST_TABLE[r][c][0]
            elif cell is Cell.O:
                h ^= _ZOBRIST_TABLE[r][c][1]
    return h


def _update_hash(h: int, row: int, col: int, player: Player) -> int:
    """Incrementally update a Zobrist hash after placing a piece.

    Args:
        h: Current hash value.
        row: Row of the new piece.
        col: Column of the new piece.
        player: Player who placed the piece.

    Returns:
        Updated hash value.
    """
    idx = 0 if player is Player.X else 1
    return h ^ _ZOBRIST_TABLE[row][col][idx]


# ---------------------------------------------------------------------------
# TT flag enum
# ---------------------------------------------------------------------------


class TTFlag(Enum):
    """Classification of a transposition table score relative to the search window.

    Attributes:
        EXACT: The stored score is the true minimax value for this position.
        LOWER: The true value is >= stored score (beta cut-off occurred).
        UPPER: The true value is <= stored score (all moves failed low).
    """

    EXACT = 0
    LOWER = 1
    UPPER = 2


# ---------------------------------------------------------------------------
# MTDf agent
# ---------------------------------------------------------------------------


class MTDf(BaseAgent):
    """MTD(f) search with a per-move local transposition table.

    MTD(f) repeatedly calls a zero-window alpha-beta (AlphaBetaWithMemory)
    with bounds [f-1, f], adjusting f after each probe, until the lower and
    upper bounds converge to a single value — the minimax score at the root.

    After convergence, the best move is recovered from the transposition
    table. If the TT has no entry for the root, a final full alpha-beta
    sweep is performed over the root's candidate moves.

    Attributes:
        depth: The fixed search depth used when no MatchConfig is provided.
        match_config: The budget configuration controlling search termination.
        _MAX_MTD_ITERATIONS: Safety cap on the number of MTD(f) probes to
            prevent infinite loops when bounds fail to converge.
    """

    _MAX_MTD_ITERATIONS = 50

    def __init__(self, depth: int = 4, match_config: MatchConfig | None = None) -> None:
        """Initialise the MTDf agent.

        Args:
            depth: Fixed search depth. Used when match_config is None or
                DEPTH_CONTROLLED.
            match_config: Budget configuration. When None, defaults to
                DEPTH_CONTROLLED with the given depth.
        """
        self.depth = depth
        if match_config is None:
            self.match_config = MatchConfig.depth_controlled(depth)
        else:
            self.match_config = match_config

    def choose_move(self, state: GameState) -> Move:
        """Select the best move using MTD(f) search.

        Args:
            state: The current game state.

        Returns:
            The chosen (row, col) move.
        """
        # Short-circuit for forced moves.
        forced = check_forced_move(state)
        if forced is not None:
            state.nodes_visited = 1
            state.max_depth_reached = 0
            state.prunings = 0
            state.compute_ebf()
            return forced

        budget = SearchBudget(self.match_config, time.perf_counter_ns())
        counters = [0, 0, 0]  # [nodes_visited, max_depth_reached, prunings]

        # Per-move transposition table: hash -> (depth, score, flag, best_move)
        tt: dict[int, tuple[int, float, TTFlag, Move | None]] = {}

        # Initial guess: heuristic evaluation of root.
        f = evaluate_position(state, state.current_player)

        # MTD(f) convergence loop.
        lower = -INF
        upper = INF
        for _ in range(self._MAX_MTD_ITERATIONS):
            if budget.exhausted(counters[0], 0):
                break

            beta = f if f == lower else f + 1 if f > lower else f

            # Clamp beta to valid range
            if beta <= lower:
                beta = lower + 1
            if beta > upper:
                beta = upper

            f = self._ab_with_memory(
                state,
                budget.max_depth(),
                beta - 1,
                beta,
                budget,
                counters,
                0,
                tt,
            )

            if f < beta:
                upper = f
            else:
                lower = f

            if lower >= upper:
                break

        # Final sweep: use TT to find the best move at root.
        root_hash = _hash_board(state.board, state.n)
        best_move = self._get_tt_best_move(tt, root_hash)

        # If TT has no move, fall back to a full search sweep.
        if best_move is None:
            candidates = Board.get_candidate_moves(state, radius=2)
            candidates = order_moves(state, candidates)
            best_move = candidates[0]  # Fallback

            best_score = -INF
            for move in candidates:
                if budget.exhausted(counters[0], 0):
                    break

                child = state.apply_move(move)
                child.result = Board.is_terminal(child.board, child.n, child.k, child.last_move)

                score = -self._ab_with_memory(
                    child,
                    budget.max_depth() - 1,
                    -INF,
                    INF,
                    budget,
                    counters,
                    1,
                    tt,
                )

                if score > best_score:
                    best_score = score
                    best_move = move

        state.nodes_visited = counters[0]
        state.max_depth_reached = counters[1]
        state.prunings = counters[2]
        state.compute_ebf()
        return best_move

    def _get_tt_best_move(
        self,
        tt: dict[int, tuple[int, float, TTFlag, Move | None]],
        board_hash: int,
    ) -> Move | None:
        """Look up the best move from the transposition table.

        Args:
            tt: The transposition table.
            board_hash: Hash of the current board.

        Returns:
            Best move if stored, None otherwise.
        """
        entry = tt.get(board_hash)
        if entry is not None:
            _, _, _, best_move = entry
            return best_move
        return None

    def _ab_with_memory(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        budget: SearchBudget,
        counters: list,
        depth_from_root: int,
        tt: dict[int, tuple[int, float, TTFlag, Move | None]],
    ) -> float:
        """Alpha-beta with transposition table (AlphaBetaWithMemory).

        Scores are from the perspective of state.current_player (negamax
        convention). The caller negates the returned score.

        Args:
            state: Current game state.
            depth: Remaining search depth.
            alpha: Lower bound.
            beta: Upper bound.
            budget: Search budget controller.
            counters: Shared [nodes_visited, max_depth_reached, prunings].
            depth_from_root: Distance from the search root.
            tt: Per-move transposition table.

        Returns:
            Best score from state.current_player's perspective.
        """
        counters[0] += 1
        counters[1] = max(counters[1], depth_from_root)

        board_hash = _hash_board(state.board, state.n)

        # TT lookup.
        original_alpha = alpha
        original_beta = beta

        entry = tt.get(board_hash)
        if entry is not None:
            tt_depth, tt_score, tt_flag, _ = entry
            if tt_depth >= depth:
                if tt_flag is TTFlag.EXACT:
                    return tt_score
                elif tt_flag is TTFlag.LOWER:
                    alpha = max(alpha, tt_score)
                elif tt_flag is TTFlag.UPPER:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    counters[2] += 1
                    return tt_score

        # Terminal or depth limit.
        if state.result != Result.IN_PROGRESS:
            score = evaluate_position(state, state.current_player)
            tt[board_hash] = (depth, score, TTFlag.EXACT, None)
            return score

        if depth == 0 or budget.exhausted(counters[0], depth_from_root):
            score = evaluate_position(state, state.current_player)
            tt[board_hash] = (depth, score, TTFlag.EXACT, None)
            return score

        candidates = Board.get_candidate_moves(state, radius=2)
        if not candidates:
            score = evaluate_position(state, state.current_player)
            tt[board_hash] = (depth, score, TTFlag.EXACT, None)
            return score

        # Prefer TT best move first.
        tt_best = self._get_tt_best_move(tt, board_hash)
        if tt_best is not None and tt_best in candidates:
            candidates.remove(tt_best)
            candidates.insert(0, tt_best)
        else:
            candidates = order_moves(state, candidates)

        best = -INF
        best_move: Move | None = candidates[0]

        for move in candidates:
            child = state.apply_move(move)
            child.result = Board.is_terminal(child.board, child.n, child.k, child.last_move)

            score = -self._ab_with_memory(
                child, depth - 1, -beta, -alpha, budget, counters, depth_from_root + 1, tt,
            )

            if score > best:
                best = score
                best_move = move

            alpha = max(alpha, best)
            if alpha >= beta:
                counters[2] += 1
                break

        # Store in TT.
        if best <= original_alpha:
            flag = TTFlag.UPPER
        elif best >= original_beta:
            flag = TTFlag.LOWER
        else:
            flag = TTFlag.EXACT
        tt[board_hash] = (depth, best, flag, best_move)

        return best

    def get_name(self) -> str:
        """Return the agent's display name including the configured depth.

        Returns:
            A string of the form "MTDf(depth=N)".
        """
        return f"MTDf(depth={self.depth})"

    def get_tier(self) -> int:
        """Return the benchmark tier for this agent.

        Returns:
            1 — classic search, no heuristic enhancements.
        """
        return 1
