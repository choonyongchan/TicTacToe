from src.core.forced_move import ForcedMove
from src.tests.test_helper import fresh_state, state_with_moves


class TestNoForcedMove:
    def test_empty_board_returns_none(self):
        assert ForcedMove.detect(fresh_state()) is None

    def test_sparse_board_returns_none(self):
        # X . .
        # . O .
        # . . .
        state = state_with_moves([(0, 0), (1, 1)])
        assert ForcedMove.detect(state) is None


class TestWinMove:
    def test_win_end_pattern_row(self):
        # X X .  ← X to move, wins at (0, 2)
        # O O .
        # . . .
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1)])
        assert ForcedMove.detect(state) == (0, 2)

    def test_win_gap_pattern_row(self):
        # X . X  ← X to move, gap at (0, 1) is the win
        # O . O
        # . . .
        # moves: X(0,0), O(1,0), X(0,2), O(1,2) → X's turn
        state = state_with_moves([(0, 0), (1, 0), (0, 2), (1, 2)])
        assert ForcedMove.detect(state) == (0, 1)

    def test_win_column(self):
        # X O .
        # X O .
        # . . .  ← X to move, wins at (2, 0)
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1)])
        assert ForcedMove.detect(state) == (2, 0)


class TestBlockMove:
    def test_blocks_opponent_end_pattern(self):
        # O O .  ← X to move, must block at (0, 2)
        # X . .
        # X . .
        # moves: X(1,0), O(0,0), X(2,0), O(0,1) → X's turn
        state = state_with_moves([(1, 0), (0, 0), (2, 0), (0, 1)])
        assert ForcedMove.detect(state) == (0, 2)

    def test_blocks_opponent_gap_pattern(self):
        # O . O  ← X to move, O has gap threat at (0, 1), must block
        # X . .
        # X . .
        # moves: X(1,0), O(0,0), X(2,0), O(0,2) → X's turn
        # X at col 0: (1,0),(2,0) but (0,0)=O → no X win
        # O at (0,0),(0,2) gap at (0,1) → block at (0,1)
        state = state_with_moves([(1, 0), (0, 0), (2, 0), (0, 2)])
        assert ForcedMove.detect(state) == (0, 1)


class TestWinPriorityOverBlock:
    def test_win_takes_priority_over_block(self):
        # X X .  ← X to move: wins at (0,2); O would win at (1,2) if not blocked
        # O O .
        # . . .
        # moves: X(0,0), O(1,0), X(0,1), O(1,1) → X's turn
        # detect must return X's win (0,2), not the block (1,2)
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1)])
        assert ForcedMove.detect(state) == (0, 2)
