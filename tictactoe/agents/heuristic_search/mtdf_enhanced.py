"""MTD(f) Enhanced — Tier 2 heuristic search.

MTD(f) (Memory-enhanced Test Driver with value f) drives a series of null-window
alpha-beta searches to converge on the minimax value. The key advantage is that
a PERSISTENT transposition table is shared across all IDDFS iterations and all
MTD(f) convergence probes, accumulating information across the entire search.

Dependency chain: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import time

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents._search_budget import SearchBudget
from tictactoe.agents._shared_utils import check_forced_move
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import INF, Move, Result, Score
from tictactoe.evaluation.heuristics import evaluate_position
from tictactoe.evaluation.move_ordering import order_moves, KillerMoveTable, HistoryTable
from tictactoe.benchmark.metrics import MatchConfig
from tictactoe.agents.heuristic_search.shared.transposition_table import (
    TranspositionTable, TTFlag,
)
from tictactoe.agents.heuristic_search.shared.threat_space_search import ThreatSpaceSearch

class MTDfEnhanced(BaseAgent):
    """MTD(f) with full heuristic enhancement stack.

    MTD(f) repeatedly calls a null-window search with progressively refined
    bounds until lower == upper (convergence). The shared, persistent TT
    is the key enabler: results from earlier probes guide later ones.

    Unlike the other Tier 2 agents, the TT is NOT cleared between IDDFS
    iterations — this is the defining feature of MTD(f).

    Enhancements:
    - Persistent transposition table (shared across iterations and probes).
    - Iterative deepening (depth loop in choose_move).
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
    ) -> None:
        """Initialise the enhanced MTD(f) agent.

        Args:
            match_config: Budget configuration. None uses default (1 s / move).
            tt_size: Transposition table size (must be a power of 2).
            use_symmetry: If True, use board symmetry to reduce TT collisions.
            use_tss: If True, run TSS before the main search.
        """
        self.match_config = match_config
        self.use_tss = use_tss
        # Persistent TT — NOT cleared between IDDFS iterations
        self._tt = TranspositionTable(tt_size, use_symmetry)
        self._killers = KillerMoveTable()
        self._history = HistoryTable()
        self._tss = ThreatSpaceSearch()
        self._last_score: Score = 0.0
        self._tss_wins_found = 0
        from tictactoe.config import get_config as _cfg, ConfigError as _CE
        try:
            _c = _cfg()
            self._id_max_depth: int = _c.search.id_max_depth
            self._max_iterations: int = _c.search.mtdf_max_iterations
            self._tss_max_depth: int = _c.search.tss_max_depth
        except _CE:
            self._id_max_depth = 1000
            self._max_iterations = 50
            self._tss_max_depth = 10

    def choose_move(self, state: GameState) -> Move:
        """Select the best move using MTD(f) with iterative deepening.

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
            tss_result = self._tss.find_forced_win(
                state, state.current_player, max_depth=self._tss_max_depth)
            if tss_result:
                self._tss_wins_found += 1
                state.nodes_visited = 1
                state.max_depth_reached = 0
                state.prunings = 0
                state.compute_ebf()
                return tss_result[0]

        # Step 3: Iterative deepening with MTD(f)
        budget = SearchBudget(self.match_config, time.perf_counter_ns())
        counters = [0, 0, 0]  # [nodes, max_depth_reached, prunings]

        # Clear killers and history (but NOT the TT — key MTD(f) feature)
        self._tt.clear()
        self._killers.clear()
        self._history.clear()
        self._last_score = evaluate_position(state, state.current_player)

        candidates = Board.get_candidate_moves(state, radius=2)
        best_move: Move | None = candidates[0] if candidates else None
        max_depth_completed = 0

        for depth in range(1, self._id_max_depth + 1):
            if budget.exhausted(counters[0], depth):
                break

            nodes_before = counters[0]
            result_score, result_move = self._search_fn(
                state, depth, -INF, INF, budget, self._tt,
                self._killers, self._history, counters,
            )

            if budget.exhausted(counters[0], depth) and depth > 1:
                nodes_added = counters[0] - nodes_before
                if nodes_added < 2:
                    break
                if result_move is not None:
                    best_move = result_move
                break

            if result_move is not None:
                best_move = result_move
            max_depth_completed = depth

        state.nodes_visited = counters[0]
        state.max_depth_reached = max(counters[1], max_depth_completed)
        state.prunings = counters[2]
        state.compute_ebf()
        return best_move

    def _search_fn(self, state: GameState, depth: int, alpha_hint: float,
                   beta_hint: float, budget: SearchBudget, tt: TranspositionTable,
                   killers: KillerMoveTable, history: HistoryTable,
                   counters: list) -> tuple[Score, Move | None]:
        """MTD(f) convergence loop for a single depth.

        Repeatedly calls a null-window (zero-width) search, adjusting bounds
        based on whether each probe returned a lower or upper bound, until the
        bounds converge (lower >= upper) or the iteration limit is reached.

        Args:
            state: Root state.
            depth: Fixed search depth for all probes.
            alpha_hint: Ignored (MTD(f) uses self._last_score as the initial f).
            beta_hint: Ignored.
            budget: Budget controller.
            tt: Shared transposition table.
            killers: Killer move table.
            history: History heuristic table.
            counters: Shared [nodes, max_depth_reached, prunings].

        Returns:
            (converged_score, best_move_from_tt) tuple.
        """
        board_hash = tt.hash_board(state.board, state.n)
        f = self._last_score
        lower: Score = -INF
        upper: Score = INF
        best_move: Move | None = None

        for _ in range(self._max_iterations):
            if budget.exhausted(counters[0], 0):
                break

            # Compute probe_beta mirroring vanilla's clamped logic:
            # after a fail-high (lower==f), probe strictly above lower to
            # allow a fail-low on the next probe and converge.
            probe_beta = f + 1 if f > lower else f
            if probe_beta <= lower:
                probe_beta = lower + 1
            if probe_beta > upper:
                probe_beta = upper
            probe_alpha = probe_beta - 1

            probe_score, _ = self._nw_search(
                state, depth, probe_alpha, probe_beta, budget, tt,
                killers, history, counters, board_hash, 0,
            )

            # Compare against probe_beta (not the old f) for fail-low/high.
            if probe_score < probe_beta:
                upper = probe_score
            else:
                lower = probe_score

            f = probe_score

            if lower >= upper:
                break

        self._last_score = f
        best_move = tt.get_best_move(board_hash)

        if best_move is None:
            candidates = Board.get_candidate_moves(state, radius=2)
            best_move = candidates[0] if candidates else None

        return f, best_move

    def _nw_search(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        budget: SearchBudget,
        tt: TranspositionTable,
        killers: KillerMoveTable,
        history: HistoryTable,
        counters: list,
        board_hash: int,
        depth_from_root: int,
    ) -> tuple[Score, Move | None]:
        """Null-window negamax search used by MTD(f) probes.

        Identical to standard negamax with alpha-beta except that it is always
        called with a null (zero-width) window from _search_fn. The TT is
        shared and persistent across probes.

        Args:
            state: Current state.
            depth: Remaining depth.
            alpha: Lower bound (typically probe_alpha in MTD(f) context).
            beta: Upper bound (typically probe_beta in MTD(f) context).
            budget: Budget controller.
            tt: Shared transposition table.
            killers: Killer move table.
            history: History heuristic.
            counters: Shared [nodes, max_depth_reached, prunings].
            board_hash: Incremental Zobrist hash.
            depth_from_root: Plies from root.

        Returns:
            (best_score, best_move) from current_player's perspective.
        """
        counters[0] += 1
        counters[1] = max(counters[1], depth_from_root)

        original_alpha = alpha

        tt_score = tt.lookup(board_hash, depth, alpha, beta)
        if tt_score is not None:
            return tt_score, tt.get_best_move(board_hash)

        if state.result != Result.IN_PROGRESS:
            return evaluate_position(state, state.current_player), None
        if depth == 0 or budget.exhausted(counters[0], depth_from_root):
            return evaluate_position(state, state.current_player), None

        candidates = Board.get_candidate_moves(state, radius=2)
        if not candidates:
            return evaluate_position(state, state.current_player), None

        tt_move = tt.get_best_move(board_hash)
        if tt_move is not None and tt_move in candidates:
            candidates.remove(tt_move)
            candidates.insert(0, tt_move)

        candidates = order_moves(state, candidates, killers, history)

        if (tt_move is not None and tt_move in candidates
                and candidates[0] != tt_move):
            candidates.remove(tt_move)
            candidates.insert(0, tt_move)

        best: Score = -INF
        best_move: Move | None = candidates[0]

        for move in candidates:
            child = state.apply_move(move)
            child.result = Board.is_terminal(
                child.board, child.n, child.k, child.last_move)

            row, col = move
            child_hash = tt.update_hash(
                board_hash, row, col, state.current_player, state.n)

            child_score, _ = self._nw_search(
                child, depth - 1, -beta, -alpha, budget, tt,
                killers, history, counters, child_hash, depth_from_root + 1,
            )
            score = -child_score

            if score > best:
                best = score
                best_move = move

            alpha = max(alpha, best)
            if alpha >= beta:
                counters[2] += 1
                killers.store(depth_from_root, move)
                history.update(move, depth)
                break

        if best <= original_alpha:
            flag = TTFlag.UPPER_BOUND
        elif best >= beta:
            flag = TTFlag.LOWER_BOUND
        else:
            flag = TTFlag.EXACT
        tt.store(board_hash, depth, best, flag, best_move)

        return best, best_move

    def get_name(self) -> str:
        """Return the agent's display name.

        Returns:
            The string "MTD(f)+Enhanced".
        """
        return "MTD(f)+Enhanced"

    def get_tier(self) -> int:
        """Return the benchmark tier for this agent.

        Returns:
            2 — heuristic-enhanced search.
        """
        return 2
