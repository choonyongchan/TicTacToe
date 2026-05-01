import pytest

from src.agents.bns_id_agent import BNSIDAgent
from src.agents.negamax_agent import NegamaxAgent
from src.core.transposition_table import TranspositionTable
from src.core.types import NEGATIVE_INFINITY
from src.tests.test_helper import (
    PUZZLE_3X3,
    PUZZLE_4X4,
    PUZZLE_4X4_BLOCK,
    PUZZLE_5X5,
    fresh_state,
    state_with_moves,
)

_EPS = 1.0 / 10  # max_depth=9 → epsilon=0.1


class TestInit:
    def test_name(self):
        assert BNSIDAgent(9).name == "BNSIDAgent"

    def test_epsilon_depth_9(self):
        assert BNSIDAgent(9)._epsilon == pytest.approx(0.1)

    def test_epsilon_depth_4(self):
        assert BNSIDAgent(4)._epsilon == pytest.approx(1.0 / 5)


class TestTerminalScore:
    def test_draw_returns_zero(self):
        state = fresh_state()
        agent = BNSIDAgent(9)
        assert agent._terminal_score(state) == 0.0

    def test_win_returns_positive(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        agent = BNSIDAgent(9)
        assert agent._terminal_score(state) > 0.0

    def test_win_score_is_depth_discounted(self):
        # X wins at history length 5: 1.0 - 0.1 * 5 = 0.5
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        agent = BNSIDAgent(9)
        assert agent._terminal_score(state) == pytest.approx(1.0 - _EPS * 5)

    def test_earlier_win_scores_higher(self):
        state_d5 = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        state_d7 = state_with_moves(
            PUZZLE_3X3.moves + ((1, 2),), PUZZLE_3X3.n, PUZZLE_3X3.k
        )
        agent = BNSIDAgent(9)
        assert agent._terminal_score(state_d5) > agent._terminal_score(state_d7)


class TestNegamaxTt:
    """_negamax_tt depth-limited behaviour — inherited from TTDepthAgent."""

    def _puzzle_state(self):
        return state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)

    def test_depth_zero_nonterminal_uses_heuristic_without_tt_store(self):
        # PUZZLE_3X3 root is not terminal; depth=0 returns heuristic and does NOT store.
        state = self._puzzle_state()
        agent = BNSIDAgent(9)
        tt = TranspositionTable()
        score = agent._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, 0, tt)
        assert -1.0 <= score <= 1.0
        assert len(tt) == 0

    def test_full_window_sufficient_depth_returns_exact(self):
        # True value = 1.0 - 0.1*7 = 0.3 (X wins at move 7 via (1,2)).
        state = self._puzzle_state()
        agent = BNSIDAgent(9)
        tt = TranspositionTable()
        score = agent._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, 9, tt)
        assert score == pytest.approx(1.0 - _EPS * 7)

    def test_fail_low_upper_bound(self):
        # alpha=0.5 > true value=0.3 → fail-low, returned score <= alpha
        state = self._puzzle_state()
        agent = BNSIDAgent(9)
        tt = TranspositionTable()
        score = agent._negamax_tt(state, 0.5, 1.0, 9, tt)
        assert score <= 0.5

    def test_fail_high_lower_bound(self):
        # beta=0.1 < true value=0.3 → fail-high, returned score >= beta
        state = self._puzzle_state()
        agent = BNSIDAgent(9)
        tt = TranspositionTable()
        score = agent._negamax_tt(state, NEGATIVE_INFINITY, 0.1, 9, tt)
        assert score >= 0.1

    def test_tt_hit_returns_consistent_value(self):
        state = self._puzzle_state()
        agent = BNSIDAgent(9)
        tt = TranspositionTable()
        first = agent._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, 9, tt)
        second = agent._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, 9, tt)
        assert first == pytest.approx(second)


class TestBnsTt:
    """_bns_tt correctness tests."""

    def _puzzle_state(self):
        return state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)

    def test_returns_best_move_puzzle_3x3(self):
        state = self._puzzle_state()
        agent = BNSIDAgent(9)
        tt = TranspositionTable()
        assert agent._bns_tt(state, -1.0, 1.0, 9, tt) == (1, 2)

    def test_returns_valid_empty_cell(self):
        state = self._puzzle_state()
        agent = BNSIDAgent(9)
        tt = TranspositionTable()
        result = agent._bns_tt(state, -1.0, 1.0, 9, tt)
        assert result is not None
        assert state.board.is_empty(*result)

    def test_tt_populated_after_search(self):
        state = self._puzzle_state()
        agent = BNSIDAgent(9)
        tt = TranspositionTable()
        agent._bns_tt(state, -1.0, 1.0, 1, tt)
        assert len(tt) > 0

    def test_window_below_true_value_still_returns_valid_move(self):
        # Window [-1, -0.5] is entirely below true value (0.3).
        # BNS will find at least one child passing the test; result is a valid cell.
        state = self._puzzle_state()
        agent = BNSIDAgent(9)
        tt = TranspositionTable()
        result = agent._bns_tt(state, -1.0, -0.5, 9, tt)
        assert result is not None
        assert state.board.is_empty(*result)


class TestActWinningMove:
    def test_takes_immediate_win_gap_row(self):
        # X . X  ← X to move, (0,1) completes row 0 (k=3)
        # O . O
        # . . .
        state = state_with_moves([(0, 0), (1, 0), (0, 2), (1, 2)])
        agent = BNSIDAgent(9)
        assert agent.act(state) == (0, 1)

    def test_takes_immediate_win_row(self):
        # X X .  → X plays (0,2) to complete row 0
        # O O .
        # . . .
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1)])
        agent = BNSIDAgent(9)
        assert agent.act(state) == (0, 2)

    def test_takes_immediate_win_column(self):
        # X O .
        # X O .
        # . . .  → X plays (2,0) to complete column 0
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1)])
        agent = BNSIDAgent(9)
        assert agent.act(state) == (2, 0)


class TestActBlockingMove:
    def test_blocks_opponent_win(self):
        # O O .  → X must block at (0,2)
        # X . .
        # X . .
        state = state_with_moves([(1, 0), (0, 0), (2, 0), (0, 1)])
        agent = BNSIDAgent(9)
        assert agent.act(state) == (0, 2)

    def test_4x4_o_blocks_x_column(self):
        state = state_with_moves(
            PUZZLE_4X4_BLOCK.moves, PUZZLE_4X4_BLOCK.n, PUZZLE_4X4_BLOCK.k
        )
        assert BNSIDAgent(PUZZLE_4X4_BLOCK.n ** 2).act(state) == PUZZLE_4X4_BLOCK.best_move


class TestActSmallTree:
    def test_picks_best_move_puzzle_3x3(self):
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = BNSIDAgent(9)
        assert agent.act(state) == PUZZLE_3X3.best_move


class TestActLargerBoards:
    def test_4x4_picks_best_move(self):
        state = state_with_moves(PUZZLE_4X4.moves, PUZZLE_4X4.n, PUZZLE_4X4.k)
        agent = BNSIDAgent(16)
        assert agent.act(state) == PUZZLE_4X4.best_move

    def test_5x5_picks_best_move(self):
        state = state_with_moves(PUZZLE_5X5.moves, PUZZLE_5X5.n, PUZZLE_5X5.k)
        agent = BNSIDAgent(25)
        assert agent.act(state) == PUZZLE_5X5.best_move


class TestActValidMove:
    def test_returns_in_bounds_move(self):
        state = fresh_state()
        agent = BNSIDAgent(9)
        row, col = agent.act(state)
        assert 0 <= row < state.board.n
        assert 0 <= col < state.board.n

    def test_returns_empty_cell(self):
        state = fresh_state()
        agent = BNSIDAgent(9)
        row, col = agent.act(state)
        assert state.board.is_empty(row, col)


class TestIterativeDeepening:
    """Tests specific to the iterative deepening structure of BNSIDAgent."""

    def test_depth_zero_nonterminal_does_not_populate_tt(self):
        # depth=0 at a non-terminal root must leave TT empty (heuristic not stored).
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = BNSIDAgent(9)
        tt = TranspositionTable()
        score = agent._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, 0, tt)
        assert -1.0 <= score <= 1.0
        assert len(tt) == 0

    def test_tt_monotonically_grows_or_stays_across_depth_iterations(self):
        # TT entries must never be removed between depth iterations.
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = BNSIDAgent(9)
        tt = TranspositionTable()
        agent._bns_tt(state, -1.0, 1.0, 1, tt)
        size_after_d1 = len(tt)
        agent._bns_tt(state, -1.0, 1.0, 2, tt)
        size_after_d2 = len(tt)
        assert size_after_d2 >= size_after_d1

    def test_best_move_at_depth_9_is_correct(self):
        # _bns_tt at depth=9 on PUZZLE_3X3 must return the known best move.
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = BNSIDAgent(9)
        tt = TranspositionTable()
        result = agent._bns_tt(state, -1.0, 1.0, 9, tt)
        assert result == (1, 2)

    def test_shallow_iteration_seeds_deeper_iteration(self):
        # Running depth=1 first seeds TT; depth=2 on same TT returns correct move.
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = BNSIDAgent(9)
        tt = TranspositionTable()
        agent._bns_tt(state, -1.0, 1.0, 1, tt)
        result = agent._bns_tt(state, -1.0, 1.0, 2, tt)
        assert result == (1, 2)

    def test_terminal_node_entry_persists_across_depth_iterations(self):
        # The winning terminal child after X plays (1,2) is stored in TT at depth=1.
        # Running depth=2 on the same TT must not evict it.
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = BNSIDAgent(9)
        tt = TranspositionTable()
        agent._bns_tt(state, -1.0, 1.0, 1, tt)
        state.apply(1, 2)
        terminal_hash = state._hash
        state.undo()
        assert tt.lookup(terminal_hash) is not None
        agent._bns_tt(state, -1.0, 1.0, 2, tt)
        assert tt.lookup(terminal_hash) is not None

    def test_act_returns_best_move_across_all_depths(self):
        # Full act() with ID: the final returned move must be PUZZLE_3X3.best_move.
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = BNSIDAgent(9)
        assert agent.act(state) == PUZZLE_3X3.best_move


class TestAgreesWithNegamax:
    """BNSIDAgent must choose identical moves to NegamaxAgent on well-defined positions."""

    def test_agrees_on_puzzle_3x3(self):
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        assert BNSIDAgent(9).act(state) == NegamaxAgent(9).act(state)

    def test_agrees_on_empty_board(self):
        # BNSIDAgent with heuristic prefers center; NegamaxAgent picks first empty cell.
        # Both are valid moves — we only verify legality.
        state = fresh_state()
        row, col = BNSIDAgent(9).act(state)
        assert 0 <= row < state.board.n
        assert 0 <= col < state.board.n
        assert state.board.is_empty(row, col)

    def test_agrees_on_puzzle_4x4(self):
        state = state_with_moves(PUZZLE_4X4.moves, PUZZLE_4X4.n, PUZZLE_4X4.k)
        assert BNSIDAgent(16).act(state) == NegamaxAgent(16).act(state)
