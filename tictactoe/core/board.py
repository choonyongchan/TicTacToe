"""Stateless board utilities for n×n Tic-Tac-Toe.

All methods are static or class methods that operate on Board2D objects or
GameState objects. The Board class holds no instance state of its own.

Dependency chain position: types → state → board → game → agents → benchmark.
"""

from __future__ import annotations

from tictactoe.core.state import GameState
from tictactoe.core.types import Board2D, Cell, Move, MoveList, Player, Result


class Board:
    """A stateless utility class providing all board operations.

    Every method is a static or class method. No Board instance should be
    created; import the class and call methods directly.
    """

    # ===================================================================
    # BOARD CREATION
    # ===================================================================

    @staticmethod
    def create(n: int) -> Board2D:
        """Create an empty n×n board.

        Args:
            n: The board dimension; must be a positive integer.

        Returns:
            An n×n grid where every cell is Cell.EMPTY.
        """
        return [[Cell.EMPTY] * n for _ in range(n)]

    # ===================================================================
    # BOARD QUERIES
    # ===================================================================

    @staticmethod
    def is_empty(board: Board2D, row: int, col: int) -> bool:
        """Check whether a specific cell is unoccupied.

        Args:
            board: The current board state.
            row: Row index (0-based).
            col: Column index (0-based).

        Returns:
            True if the cell contains Cell.EMPTY.
        """
        return board[row][col] is Cell.EMPTY

    @staticmethod
    def is_full(board: Board2D) -> bool:
        """Check whether every cell on the board has been claimed.

        Args:
            board: The current board state.

        Returns:
            True if no Cell.EMPTY cells remain.
        """
        return all(cell is not Cell.EMPTY for row in board for cell in row)

    @staticmethod
    def get_cell(board: Board2D, row: int, col: int) -> Cell:
        """Retrieve the contents of a cell.

        Args:
            board: The current board state.
            row: Row index (0-based).
            col: Column index (0-based).

        Returns:
            The Cell value at (row, col).
        """
        return board[row][col]

    @staticmethod
    def count_pieces(board: Board2D, player: Player) -> int:
        """Count how many cells a player has claimed.

        Args:
            board: The current board state.
            player: The player whose pieces are counted.

        Returns:
            The number of cells occupied by player.
        """
        target = player.to_cell()
        return sum(cell is target for row in board for cell in row)

    @staticmethod
    def get_all_empty_cells(board: Board2D) -> MoveList:
        """Return all unoccupied cells as (row, col) pairs.

        Args:
            board: The current board state.

        Returns:
            A list of (row, col) moves for every empty cell, in row-major order.
        """
        return [
            (row_idx, col_idx)
            for row_idx, row in enumerate(board)
            for col_idx, cell in enumerate(row)
            if cell is Cell.EMPTY
        ]

    # ===================================================================
    # WIN DETECTION
    # ===================================================================

    @staticmethod
    def _check_direction(
        board: Board2D,
        row: int,
        col: int,
        delta_row: int,
        delta_col: int,
        player_cell: Cell,
        k: int,
    ) -> int:
        """Count consecutive player cells in one direction from (row, col).

        Starts one step away from (row, col) in the direction (delta_row,
        delta_col) and counts cells belonging to player_cell until a
        different cell or a board boundary is reached.

        Args:
            board: The current board state.
            row: Starting row index.
            col: Starting column index.
            delta_row: Row step direction (-1, 0, or 1).
            delta_col: Column step direction (-1, 0, or 1).
            player_cell: The Cell value to match.
            k: Maximum line length needed for a win (used as the walk limit).

        Returns:
            The number of consecutive matching cells in the given direction.
        """
        n = len(board)
        count = 0
        current_row = row + delta_row
        current_col = col + delta_col

        while (
            0 <= current_row < n
            and 0 <= current_col < n
            and board[current_row][current_col] is player_cell
            and count < k
        ):
            count += 1
            current_row += delta_row
            current_col += delta_col

        return count

    @classmethod
    def check_win(
        cls, board: Board2D, n: int, k: int, last_move: Move
    ) -> Player | None:
        """Check whether the last move created a winning line.

        Only examines lines that pass through last_move, making this suitable
        for incremental win detection during search (O(k) instead of O(n²)).

        Args:
            board: The current board state.
            n: Board dimension.
            k: Number of consecutive cells needed to win.
            last_move: The (row, col) of the move just played.

        Returns:
            The winning Player if a k-in-a-row is found, otherwise None.
        """
        row, col = last_move
        player_cell = board[row][col]

        if player_cell is Cell.EMPTY:
            return None

        axes = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal
            (1, -1),  # anti-diagonal
        ]

        for delta_row, delta_col in axes:
            forward = cls._check_direction(
                board, row, col, delta_row, delta_col, player_cell, k
            )
            backward = cls._check_direction(
                board, row, col, -delta_row, -delta_col, player_cell, k
            )
            # +1 for the cell at last_move itself.
            if forward + backward + 1 >= k:
                return Player.X if player_cell is Cell.X else Player.O

        return None

    @classmethod
    def check_win_full(cls, board: Board2D, n: int, k: int) -> Player | None:
        """Scan the entire board for a winning line.

        This is an O(n² · k) full-board scan intended only for validation
        in tests. Use check_win for incremental detection during search.

        Args:
            board: The current board state.
            n: Board dimension.
            k: Number of consecutive cells needed to win.

        Returns:
            The winning Player if any k-in-a-row is found, otherwise None.
        """
        for row in range(n):
            for col in range(n):
                if board[row][col] is Cell.EMPTY:
                    continue
                winner = cls.check_win(board, n, k, (row, col))
                if winner is not None:
                    return winner
        return None

    @classmethod
    def is_terminal(
        cls, board: Board2D, n: int, k: int, last_move: Move | None
    ) -> Result:
        """Determine the current game outcome.

        Args:
            board: The current board state.
            n: Board dimension.
            k: Number of consecutive cells needed to win.
            last_move: The most recent move, or None if no moves have been made.

        Returns:
            Result.X_WINS or Result.O_WINS if a winner exists,
            Result.DRAW if the board is full with no winner,
            Result.IN_PROGRESS otherwise.
        """
        if last_move is not None:
            winner = cls.check_win(board, n, k, last_move)
            if winner is Player.X:
                return Result.X_WINS
            if winner is Player.O:
                return Result.O_WINS

        if cls.is_full(board):
            return Result.DRAW

        return Result.IN_PROGRESS

    # ===================================================================
    # MOVE UTILITIES
    # ===================================================================

    @staticmethod
    def _chebyshev_neighbours(
        n: int, row: int, col: int, radius: int
    ) -> MoveList:
        """Return all in-bounds cells within Chebyshev distance of (row, col).

        Args:
            n: Board dimension.
            row: Centre row index.
            col: Centre column index.
            radius: Maximum Chebyshev distance to include.

        Returns:
            A list of (r, c) pairs that are within radius steps of (row, col).
        """
        neighbours: MoveList = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                neighbour_row = row + dr
                neighbour_col = col + dc
                if 0 <= neighbour_row < n and 0 <= neighbour_col < n:
                    neighbours.append((neighbour_row, neighbour_col))
        return neighbours

    @classmethod
    def get_candidate_moves(
        cls, state: GameState, radius: int = 2
    ) -> MoveList:
        """Return empty cells near occupied cells within a Chebyshev radius.

        This heuristic reduces the branching factor for larger boards while
        retaining all strategically relevant moves in typical positions.

        Fallback rules:
        - If the board is empty, return the centre cell (or all cells if n=1).
        - If no empty cell lies within radius of any occupied cell, return
          all empty cells (safety fallback).

        Args:
            state: The current game state.
            radius: Chebyshev distance limit.

        Returns:
            A deduplicated list of candidate (row, col) moves.
        """
        board = state.board
        n = state.n

        occupied = [
            (row, col)
            for row in range(n)
            for col in range(n)
            if board[row][col] is not Cell.EMPTY
        ]

        if not occupied:
            # Board is empty — prefer the centre.
            centre = n // 2
            return [(centre, centre)]

        candidate_set: set[Move] = set()
        for occ_row, occ_col in occupied:
            for neighbour in cls._chebyshev_neighbours(n, occ_row, occ_col, radius):
                n_row, n_col = neighbour
                if board[n_row][n_col] is Cell.EMPTY:
                    candidate_set.add((n_row, n_col))

        if not candidate_set:
            return cls.get_all_empty_cells(board)

        return list(candidate_set)

    @staticmethod
    def _would_win(
        board: Board2D, n: int, k: int, row: int, col: int, player: Player
    ) -> bool:
        """Check whether placing player's piece at (row, col) wins immediately.

        Args:
            board: The current board state.
            n: Board dimension.
            k: Number of consecutive cells needed to win.
            row: Target row index.
            col: Target column index.
            player: The player attempting the move.

        Returns:
            True if the move results in a win for player.
        """
        # Temporarily place the piece.
        board[row][col] = player.to_cell()
        winner = Board.check_win(board, n, k, (row, col))
        # Undo the temporary placement.
        board[row][col] = Cell.EMPTY
        return winner is player

    @classmethod
    def get_winning_move(
        cls, board: Board2D, n: int, k: int, player: Player
    ) -> Move | None:
        """Return the first empty cell where player can win immediately.

        Args:
            board: The current board state.
            n: Board dimension.
            k: Number of consecutive cells needed to win.
            player: The player to check winning moves for.

        Returns:
            A (row, col) winning move, or None if no immediate win exists.
        """
        for row in range(n):
            for col in range(n):
                if board[row][col] is Cell.EMPTY:
                    if cls._would_win(board, n, k, row, col, player):
                        return (row, col)
        return None

    @classmethod
    def get_blocking_move(
        cls, board: Board2D, n: int, k: int, player: Player
    ) -> Move | None:
        """Return a move that prevents the opponent from winning immediately.

        Args:
            board: The current board state.
            n: Board dimension.
            k: Number of consecutive cells needed to win.
            player: The player who needs to block (checks opponent's threats).

        Returns:
            A (row, col) blocking move, or None if no immediate threat exists.
        """
        opponent = player.opponent()
        return cls.get_winning_move(board, n, k, opponent)

    # ===================================================================
    # DISPLAY
    # ===================================================================

    @staticmethod
    def render(board: Board2D, n: int, last_move: Move | None = None) -> str:
        """Build a human-readable string representation of the board.

        The last move is highlighted with square brackets (e.g., [X]).
        Row and column indices are printed along the borders.

        Args:
            board: The current board state.
            n: Board dimension.
            last_move: Optional (row, col) of the last move to highlight.

        Returns:
            A multi-line string suitable for printing to a terminal.
        """
        cell_symbols = {Cell.EMPTY: ".", Cell.X: "X", Cell.O: "O"}

        col_header = "    " + "  ".join(str(col) for col in range(n))
        separator = "   " + "---" * n

        lines = [col_header, separator]
        for row in range(n):
            row_cells = []
            for col in range(n):
                symbol = cell_symbols[board[row][col]]
                if last_move is not None and (row, col) == last_move:
                    row_cells.append(f"[{symbol}]")
                else:
                    row_cells.append(f" {symbol} ")
            lines.append(f"{row} |{'|'.join(row_cells)}|")

        return "\n".join(lines)
