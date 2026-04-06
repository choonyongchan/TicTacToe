"""MinimaxAB Enhanced — Tier 2 heuristic search.

Extends the classic alpha-beta minimax agent with a full heuristic
enhancement stack:

1. Forced move short-circuit: immediately returns any winning or blocking
   move detected by check_forced_move without entering the search tree.
2. Threat Space Search (TSS) pre-search: detects forced wins via threat
   sequences before running the main search. Exploits domain knowledge to
   find wins much faster than alpha-beta.
3. Iterative Deepening DFS (IDDFS) with aspiration windows: searches to
   increasing depths, returning the last fully-completed iteration's result.
   Aspiration windows narrow the search around the previous score estimate.
4. Transposition table (two-tier replacement): caches evaluated positions
   to avoid re-searching transpositions. Cleared at the start of each move.
5. TT move ordering: the TT best move is always placed first in the candidate
   list, ahead of the killer/history/static ordering.
6. Killer move heuristic: two killer moves per ply are tried early.
7. History heuristic: moves that caused cut-offs at higher depths are
   ordered first.

The core search is true minimax with MAX/MIN alternation. The maximising
player is fixed at the root and all scores throughout the tree are relative
to that player. MAX nodes maximise the score and update alpha; MIN nodes
minimise the score and update beta. No score negation is used.

Dependency chain: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import time

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents._search_budget import SearchBudget
from tictactoe.agents._shared_utils import check_forced_move
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import INF, Move, Player, Result, Score
from tictactoe.evaluation.heuristics import evaluate_position
from tictactoe.evaluation.move_ordering import order_moves, KillerMoveTable, HistoryTable
from tictactoe.benchmark.metrics import MatchConfig
from tictactoe.agents.heuristic_search.shared.transposition_table import (
    TranspositionTable, TTFlag,
)
from tictactoe.agents.heuristic_search.shared.iterative_deepening import (
    IterativeDeepeningWrapper,
)
from tictactoe.agents.heuristic_search.shared.threat_space_search import ThreatSpaceSearch


class MinimaxABEnhanced(BaseAgent):
    """MinimaxAB with full heuristic enhancement stack.

    Uses true minimax with MAX/MIN alternation: the root player is fixed as
    the maximising player and all scores throughout the tree are relative to
    that player. MAX nodes maximise the score; MIN nodes minimise it.

    Enhancements over plain alpha-beta:
    - Transposition table with two-tier replacement.
    - Iterative deepening with aspiration windows.
    - Killer move heuristic.
    - History heuristic.
    - Threat Space Search pre-search for forced wins.
    """

    def __init__(
        self,
        match_config: MatchConfig | None = None,
        tt_size: int = 2**20,
        use_symmetry: bool = False,
        use_tss: bool = True,
        use_aspiration: bool = True,
        aspiration_delta: float = 50.0,
    ) -> None:
        """Initialise the enhanced agent.

        Args:
            match_config: Budget configuration. None uses default (1 s / move).
            tt_size: Transposition table size (must be a power of 2).
            use_symmetry: If True, use board symmetry to reduce TT collisions.
            use_tss: If True, run TSS before the main search.
            use_aspiration: If True, use aspiration windows in IDDFS.
            aspiration_delta: Half-width of the aspiration window.
        """
        self.match_config = match_config
        self.use_tss = use_tss
        self._tt = TranspositionTable(tt_size, use_symmetry)
        self._killers = KillerMoveTable()
        self._history = HistoryTable()
        self._tss = ThreatSpaceSearch()
        self._id = IterativeDeepeningWrapper(
            self._search_fn, use_aspiration, aspiration_delta)
        self._tss_wins_found = 0

    def choose_move(self, state: GameState) -> Move:
        """Select the best move using enhanced minimax search.

        Args:
            state: The current game state.

        Returns:
            The chosen (row, col) move.
        """
        # Step 1: Forced move check
        forced = check_forced_move(state)
        if forced is not None:
            state.nodes_visited = 1
            state.max_depth_reached = 0
            state.prunings = 0
            state.compute_ebf()
            return forced

        # Step 2: TSS pre-search
        if self.use_tss:
            tss_result = self._tss.find_forced_win(state, state.current_player)
            if tss_result:
                self._tss_wins_found += 1
                state.nodes_visited = 1
                state.max_depth_reached = 0
                state.prunings = 0
                state.compute_ebf()
                return tss_result[0]

        # Step 3: IDDFS
        budget = SearchBudget(self.match_config, time.perf_counter_ns())
        counters = [0, 0, 0]  # [nodes, max_depth_reached, prunings]

        # Clear state between moves
        self._tt.clear()
        self._killers.clear()
        self._history.clear()

        best_move, _, max_depth = self._id.run(
            state, budget, self._tt, self._killers, self._history, counters)

        state.nodes_visited = counters[0]
        state.max_depth_reached = max(counters[1], max_depth)
        state.prunings = counters[2]
        state.compute_ebf()
        return best_move

    def _search_fn(self, state: GameState, depth: int, alpha: float, beta: float,
                   budget: SearchBudget, tt: TranspositionTable,
                   killers: KillerMoveTable, history: HistoryTable,
                   counters: list) -> tuple[Score, Move | None]:
        """Entry point for one IDDFS iteration. Returns (score, best_move).

        The root player is always the maximising player. All scores returned
        from the tree are relative to this player.

        Args:
            state: Root state for this iteration.
            depth: Maximum depth for this iteration.
            alpha: Alpha bound (best score MAX can guarantee).
            beta: Beta bound (best score MIN can guarantee).
            budget: Search budget controller.
            tt: Transposition table.
            killers: Killer move table.
            history: History heuristic table.
            counters: Shared [nodes, max_depth_reached, prunings].

        Returns:
            (score, best_move_at_root) tuple.
        """
        board_hash = tt.hash_board(state.board, state.n)
        maximising_player = state.current_player

        score, _ = self._minimax(
            state, depth, alpha, beta, True, budget, tt,
            killers, history, counters, board_hash, 0, maximising_player, True,
        )
        # Get best move from TT
        best_move_at_root = tt.get_best_move(board_hash)

        # Fallback to first candidate
        if best_move_at_root is None:
            candidates = Board.get_candidate_moves(state, radius=2)
            if candidates:
                best_move_at_root = candidates[0]

        return score, best_move_at_root

    def _minimax(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        is_maximising: bool,
        budget: SearchBudget,
        tt: TranspositionTable,
        killers: KillerMoveTable,
        history: HistoryTable,
        counters: list,
        board_hash: int,
        depth_from_root: int,
        maximising_player: Player,
        is_root: bool = False,
    ) -> tuple[Score, Move | None]:
        """Enhanced minimax search with alpha-beta pruning and TT.

        All scores are from maximising_player's perspective (true minimax
        convention). MAX nodes maximise the score and update alpha; MIN nodes
        minimise the score and update beta. No score negation is used.

        Args:
            state: Current state.
            depth: Remaining depth.
            alpha: Best score MAX can currently guarantee (lower bound).
            beta: Best score MIN can currently guarantee (upper bound).
            is_maximising: True at MAX nodes (root player's turn); False at
                MIN nodes (opponent's turn).
            budget: Budget controller.
            tt: Transposition table.
            killers: Killer move table.
            history: History heuristic.
            counters: Shared [nodes, max_depth_reached, prunings].
            board_hash: Incremental Zobrist hash of the current board.
            depth_from_root: Distance from the search root (for killer indexing).
            maximising_player: The root player; all scores are relative to this
                player and never change throughout the search.
            is_root: True only at the root node (skips TT cutoff there).

        Returns:
            (best_score, best_move) where best_score is from
            maximising_player's perspective.
        """
        counters[0] += 1
        counters[1] = max(counters[1], depth_from_root)

        original_alpha = alpha
        original_beta = beta

        # TT lookup (skip at root to always get a best_move)
        tt_score = tt.lookup(board_hash, depth, alpha, beta)
        if tt_score is not None and not is_root:
            return tt_score, tt.get_best_move(board_hash)

        if state.result != Result.IN_PROGRESS:
            return evaluate_position(state, maximising_player), None
        if depth == 0 or budget.exhausted(counters[0], depth_from_root):
            return evaluate_position(state, maximising_player), None

        candidates = Board.get_candidate_moves(state, radius=2)
        if not candidates:
            return evaluate_position(state, maximising_player), None

        # TT move ordering: place TT best move first
        tt_move = tt.get_best_move(board_hash)
        if tt_move is not None and tt_move in candidates:
            candidates.remove(tt_move)
            candidates.insert(0, tt_move)

        candidates = order_moves(state, candidates, killers, history)

        # Re-insert TT move at front after ordering if displaced
        if (tt_move is not None and tt_move in candidates
                and candidates[0] != tt_move):
            candidates.remove(tt_move)
            candidates.insert(0, tt_move)

        best_move: Move | None = candidates[0]

        if is_maximising:
            best: Score = -INF
            for move in candidates:
                child = state.apply_move(move)
                child.result = Board.is_terminal(
                    child.board, child.n, child.k, child.last_move)

                row, col = move
                child_hash = tt.update_hash(
                    board_hash, row, col, state.current_player, state.n)

                child_score, _ = self._minimax(
                    child, depth - 1, alpha, beta, False, budget, tt,
                    killers, history, counters, child_hash, depth_from_root + 1,
                    maximising_player, False,
                )

                if child_score > best:
                    best = child_score
                    best_move = move

                alpha = max(alpha, best)
                if alpha >= beta:
                    counters[2] += 1
                    killers.store(depth_from_root, move)
                    history.update(move, depth)
                    break

            # MAX node flag: relative to original alpha and (unchanged) beta
            if best <= original_alpha:
                flag = TTFlag.UPPER_BOUND
            elif best >= beta:
                flag = TTFlag.LOWER_BOUND
            else:
                flag = TTFlag.EXACT

        else:  # Minimising
            best = INF
            for move in candidates:
                child = state.apply_move(move)
                child.result = Board.is_terminal(
                    child.board, child.n, child.k, child.last_move)

                row, col = move
                child_hash = tt.update_hash(
                    board_hash, row, col, state.current_player, state.n)

                child_score, _ = self._minimax(
                    child, depth - 1, alpha, beta, True, budget, tt,
                    killers, history, counters, child_hash, depth_from_root + 1,
                    maximising_player, False,
                )

                if child_score < best:
                    best = child_score
                    best_move = move

                beta = min(beta, best)
                if alpha >= beta:
                    counters[2] += 1
                    killers.store(depth_from_root, move)
                    history.update(move, depth)
                    break

            # MIN node flag: relative to (unchanged) alpha and original beta
            if best >= original_beta:
                flag = TTFlag.LOWER_BOUND
            elif best <= alpha:
                flag = TTFlag.UPPER_BOUND
            else:
                flag = TTFlag.EXACT

        tt.store(board_hash, depth, best, flag, best_move)
        return best, best_move

    def get_name(self) -> str:
        """Return the agent's display name.

        Returns:
            The string "MinimaxAB+Enhanced".
        """
        return "MinimaxAB+Enhanced"

    def get_tier(self) -> int:
        """Return the benchmark tier for this agent.

        Returns:
            2 — heuristic-enhanced search.
        """
        return 2
