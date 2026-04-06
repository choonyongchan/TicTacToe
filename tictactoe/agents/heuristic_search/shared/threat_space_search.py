"""Threat Space Search (TSS) for forced win detection.

Threat Space Search is a domain-specific algorithm that restricts the search
to sequences of threats (gain moves) and forced defensive responses (cost
moves), rather than exploring the full game tree.

Algorithm overview:
    1. Generate all "gain" moves — moves that create a threat (open or
       half-open three/four in a k-in-a-row game).
    2. For each gain move, determine the defender's forced "cost" moves —
       moves that must be played to prevent an immediate loss.
    3. Recurse on the resulting position (with depth decremented by 2:
       one ply for attacker, one for defender).
    4. A forced win is confirmed only when a gain move defeats ALL of the
       defender's possible cost responses.

TSS is typically run as a pre-search step before the main alpha-beta search.
It is dramatically faster than alpha-beta on positions where a forced win
exists, because the branching factor is the number of threats rather than
the number of candidate moves.

Threat strength hierarchy (ThreatType enum, strongest to weakest):
    OPEN_FOUR > HALF_OPEN_FOUR > OPEN_THREE > HALF_OPEN_THREE

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

from enum import Enum

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Move, MoveList, Player, Result
from tictactoe.evaluation.heuristics import count_open_threats, precompute_line_indices


class ThreatType(Enum):
    """Classification of a threat by its strategic strength.

    Higher enum values indicate stronger threats that are harder to defend
    against and are therefore searched first.

    Attributes:
        OPEN_FOUR: k-1 pieces in a line with both ends open (unblocked).
            One move from winning with two ways to complete the k-in-a-row.
        HALF_OPEN_FOUR: k-1 pieces in a line with one end open.
            One move from winning via a single completion square.
        OPEN_THREE: k-2 pieces in a line with both ends open.
            Two moves from winning; can become an open four next turn.
        HALF_OPEN_THREE: k-2 pieces in a line with one end open.
            Weaker threat; only useful if the opponent fails to respond.
    """

    OPEN_FOUR = 4
    HALF_OPEN_FOUR = 3
    OPEN_THREE = 2
    HALF_OPEN_THREE = 1


class ThreatSpaceSearch:
    """Domain-specific search for forced wins via threat sequences.

    Explores only positions reachable by playing threats (gain moves) and
    their forced defensive responses (cost moves). Because the branching
    factor is bounded by the number of threats rather than all legal moves,
    TSS can prove forced wins orders of magnitude faster than alpha-beta.

    The search is complete for the threat types it considers: if a forced
    win exists via the OPEN_FOUR, HALF_OPEN_FOUR, OPEN_THREE, and
    HALF_OPEN_THREE threat hierarchy, TSS will find it within MAX_TSS_DEPTH
    plies. It does not, however, prove that no forced win exists — a None
    result merely means no such win was found within the depth limit.

    Attributes:
        MAX_TSS_DEPTH: Maximum number of plies explored. Each attacker move
            plus each defender response counts as two plies, so a depth of 10
            allows up to 5 attacker moves.
    """

    MAX_TSS_DEPTH = 10

    def find_forced_win(self, state: GameState, player: Player,
                        max_depth: int | None = None) -> list[Move] | None:
        """Return winning move sequence if forced win found, else None.

        Args:
            state: Current game state.
            player: The attacking player seeking a forced win.
            max_depth: Maximum threat sequence length to explore (default 10).

        Returns:
            An ordered list of moves forming a forced win, or None.
        """
        if max_depth is None:
            max_depth = self.MAX_TSS_DEPTH
        return self._tss(state, player, max_depth, [])

    def _tss(self, state: GameState, player: Player, depth_remaining: int,
             sequence: list[Move]) -> list[Move] | None:
        """Recursive TSS implementation.

        Args:
            state: Current game state.
            player: The attacking player.
            depth_remaining: How many more plies we may explore.
            sequence: Moves accumulated so far in this branch.

        Returns:
            Full winning sequence, or None.
        """
        if depth_remaining <= 0:
            return None

        threats = self._get_threat_moves(state, player)
        if not threats:
            return None

        for gain_move, threat_type in threats:
            next_state = state.apply_move(gain_move)
            next_state.result = Board.is_terminal(
                next_state.board, next_state.n, next_state.k, gain_move
            )

            if next_state.result != Result.IN_PROGRESS:
                return sequence + [gain_move]  # Immediate win

            # Get opponent's forced responses
            costs = self._get_cost_moves(next_state, gain_move, player)

            if not costs:
                return sequence + [gain_move]  # No defence needed

            # Try each cost; win only if we beat ALL defenses
            all_beaten = True
            for cost_move in costs:
                after_cost = next_state.apply_move(cost_move)
                after_cost.result = Board.is_terminal(
                    after_cost.board, after_cost.n, after_cost.k, cost_move
                )
                if after_cost.result != Result.IN_PROGRESS:
                    all_beaten = False
                    break
                result = self._tss(after_cost, player, depth_remaining - 2,
                                   sequence + [gain_move, cost_move])
                if result is None:
                    all_beaten = False
                    break

            if all_beaten:
                return sequence + [gain_move]

        return None

    def _get_threat_moves(self, state: GameState,
                          player: Player) -> list[tuple[Move, ThreatType]]:
        """Return (move, threat_type) for moves creating threats.

        Args:
            state: Current game state.
            player: The attacking player.

        Returns:
            List of (move, ThreatType) sorted by threat strength descending.
        """
        threats: list[tuple[Move, ThreatType]] = []
        board = state.board
        n = state.n

        for row in range(n):
            for col in range(n):
                if board[row][col] is not Cell.EMPTY:
                    continue
                threat_type = self._classify_threat(state, (row, col), player)
                if threat_type is not None:
                    threats.append(((row, col), threat_type))

        # Sort by threat strength (higher enum value = stronger)
        threats.sort(key=lambda x: x[1].value, reverse=True)
        return threats

    def _classify_threat(self, state: GameState, move: Move,
                          player: Player) -> ThreatType | None:
        """Classify the threat type of a move.

        Temporarily places the piece, checks for threat patterns, then undoes.

        Args:
            state: Current game state.
            move: The candidate move.
            player: The player making the move.

        Returns:
            The strongest ThreatType created, or None if no threat.
        """
        board = state.board
        n = state.n
        k = state.k
        row, col = move

        # Temporarily place the piece
        board[row][col] = player.to_cell()

        threat_type = None

        # Check for open/half-open fours (k-1 pieces in line)
        if k >= 2:
            open_fours = self._count_open_lines(
                board, n, k, player, k - 1, require_open_ends=2)
            half_open_fours = self._count_open_lines(
                board, n, k, player, k - 1, require_open_ends=1)
            if open_fours > 0:
                threat_type = ThreatType.OPEN_FOUR
            elif half_open_fours > 0:
                threat_type = ThreatType.HALF_OPEN_FOUR

        if threat_type is None and k >= 3:
            open_threes = self._count_open_lines(
                board, n, k, player, k - 2, require_open_ends=2)
            half_open_threes = self._count_open_lines(
                board, n, k, player, k - 2, require_open_ends=1)
            if open_threes > 0:
                threat_type = ThreatType.OPEN_THREE
            elif half_open_threes > 0:
                threat_type = ThreatType.HALF_OPEN_THREE

        board[row][col] = Cell.EMPTY  # Undo
        return threat_type

    def _count_open_lines(self, board: list, n: int, k: int, player: Player,
                           count: int, require_open_ends: int) -> int:
        """Count lines with exactly `count` player pieces and required open ends.

        Args:
            board: Current board state.
            n: Board dimension.
            k: Winning line length.
            player: The player whose pieces are counted.
            count: Exact number of player pieces required in the line.
            require_open_ends: Minimum number of open (empty) cells in the line.

        Returns:
            Number of lines satisfying the criteria.
        """
        player_cell = player.to_cell()
        opponent_cell = player.opponent().to_cell()
        total = 0

        for line_coords in precompute_line_indices(n, k):
            cells = [board[r][c] for r, c in line_coords]
            player_count = sum(1 for c in cells if c is player_cell)
            opp_count = sum(1 for c in cells if c is opponent_cell)

            if player_count == count and opp_count == 0:
                # Count open ends (empty cells in the line)
                empty_count = sum(1 for c in cells if c is Cell.EMPTY)
                open_ends = min(empty_count, 2)  # Simplification
                if open_ends >= require_open_ends:
                    total += 1

        return total

    def _get_cost_moves(self, state: GameState, gain_move: Move,
                         attacker: Player) -> MoveList:
        """Return defender's forced blocking moves.

        Args:
            state: Game state AFTER the attacker's gain move.
            gain_move: The move just played by the attacker.
            attacker: The attacking player.

        Returns:
            List of moves the defender is forced to consider.
        """
        defender = attacker.opponent()
        costs: MoveList = []

        # The defender must block immediate wins
        win_move = Board.get_winning_move(state.board, state.n, state.k, attacker)
        if win_move is not None:
            block_move = Board.get_blocking_move(
                state.board, state.n, state.k, defender)
            if block_move is not None:
                costs.append(block_move)
            return costs

        # Also check if defender has a winning move (which also serves as a block)
        defender_win = Board.get_winning_move(
            state.board, state.n, state.k, defender)
        if defender_win is not None:
            costs.append(defender_win)

        # Block open fours
        board = state.board
        n = state.n
        k = state.k

        for row in range(n):
            for col in range(n):
                if board[row][col] is not Cell.EMPTY:
                    continue
                # Check if this blocks an attacker's open four
                board[row][col] = attacker.to_cell()
                open_four = self._count_open_lines(
                    board, n, k, attacker, k - 1, require_open_ends=2)
                board[row][col] = Cell.EMPTY
                if open_four > 0:
                    if (row, col) not in costs:
                        costs.append((row, col))

        return costs
