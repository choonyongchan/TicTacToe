import pytest

from src.agents.mtdf_id_agent import MTDfIDAgent
from src.agents.mtdf_agent import MTDfAgent
from src.agents.negamax_agent import NegamaxAgent
from src.core.transposition_table import TranspositionTable
from src.core.types import NEGATIVE_INFINITY
from src.tests.test_helper import (
    PUZZLE_3X3,
    PUZZLE_4X4,
    PUZZLE_5X5,
    fresh_state,
    state_with_moves,
)

_EPS = 1.0 / 10  # max_depth=9 → epsilon=0.1


class TestInit:
    def test_name(self):
        assert MTDfIDAgent(9).name == "MTDfIDAgent"

    def test_epsilon_depth_9(self):
        assert MTDfIDAgent(9)._epsilon == pytest.approx(0.1)


class TestTerminalScore:
    def test_draw_returns_zero(self):
        state = fresh_state()
        agent = MTDfIDAgent(9)
        assert agent._terminal_score(state) == 0.0

    def test_win_returns_positive(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        agent = MTDfIDAgent(9)
        assert agent._terminal_score(state) > 0.0

    def test_win_score_is_depth_discounted(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        agent = MTDfIDAgent(9)
        assert agent._terminal_score(state) == pytest.approx(1.0 - _EPS * 5)

    def test_earlier_win_scores_higher(self):
        state_d5 = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        state_d7 = state_with_moves(
            PUZZLE_3X3.moves + ((1, 2),), PUZZLE_3X3.n, PUZZLE_3X3.k
        )
        agent = MTDfIDAgent(9)
        assert agent._terminal_score(state_d5) > agent._terminal_score(state_d7)


class TestMtdfId:
    """_mtdf with explicit depth must converge to the exact value.

    PUZZLE_3X3 true value = 1.0 - 0.1*7 = 0.3 (X wins at (1,2), history length 7).
    depth=1 is sufficient: (1,2) is an immediate terminal win from puzzle position.
    """

    def _puzzle_state(self):
        return state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)

    def test_converges_at_depth_1_from_zero_guess(self):
        state = self._puzzle_state()
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        score = agent._mtdf(state, 0.0, 1, tt)
        assert score == pytest.approx(1.0 - _EPS * 7)

    def test_converges_at_depth_1_from_high_guess(self):
        state = self._puzzle_state()
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        score = agent._mtdf(state, 1.0, 1, tt)
        assert score == pytest.approx(1.0 - _EPS * 7)

    def test_converges_at_depth_1_from_low_guess(self):
        state = self._puzzle_state()
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        score = agent._mtdf(state, -1.0, 1, tt)
        assert score == pytest.approx(1.0 - _EPS * 7)

    def test_tt_populated_after_search(self):
        state = self._puzzle_state()
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        agent._mtdf(state, 0.0, 1, tt)
        assert len(tt) > 0


class TestNegamaxTtDepthLimited:
    """_negamax_tt depth-limited behaviour."""

    def _puzzle_state(self):
        return state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)

    def test_depth_zero_nonterminal_uses_heuristic_without_tt_store(self):
        # PUZZLE_3X3 root is not terminal; depth=0 must return the heuristic value
        # (non-zero since there are pieces on the board) and NOT store in TT.
        state = self._puzzle_state()
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        score = agent._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, 0, tt)
        assert -1.0 <= score <= 1.0
        assert len(tt) == 0

    def test_full_window_sufficient_depth_returns_exact(self):
        state = self._puzzle_state()
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        score = agent._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, 9, tt)
        assert score == pytest.approx(1.0 - _EPS * 7)

    def test_fail_low_upper_bound(self):
        # alpha=0.5 > true value=0.3 → fail-low, score <= alpha
        state = self._puzzle_state()
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        score = agent._negamax_tt(state, 0.5, 1.0, 9, tt)
        assert score <= 0.5

    def test_fail_high_lower_bound(self):
        # beta=0.1 < true value=0.3 → fail-high (beta cutoff), score >= beta
        state = self._puzzle_state()
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        score = agent._negamax_tt(state, NEGATIVE_INFINITY, 0.1, 9, tt)
        assert score >= 0.1

    def test_tt_hit_avoids_research(self):
        state = self._puzzle_state()
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        first = agent._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, 9, tt)
        second = agent._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, 9, tt)
        assert first == pytest.approx(second)


class TestActWinningMove:
    def test_takes_immediate_win_row(self):
        # X X .  → X plays (0, 2) to complete row 0
        # O O .
        # . . .
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1)])
        agent = MTDfIDAgent(9)
        assert agent.act(state) == (0, 2)

    def test_takes_immediate_win_column(self):
        # X O .
        # X O .
        # . . .  → X plays (2, 0) to complete column 0
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1)])
        agent = MTDfIDAgent(9)
        assert agent.act(state) == (2, 0)


class TestActBlockingMove:
    def test_blocks_opponent_win(self):
        # O O .  → X must block at (0, 2)
        # X . .
        # X . .
        state = state_with_moves([(1, 0), (0, 0), (2, 0), (0, 1)])
        agent = MTDfIDAgent(9)
        assert agent.act(state) == (0, 2)


class TestActSmallTree:
    def test_picks_best_move_puzzle_3x3(self):
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = MTDfIDAgent(9)
        assert agent.act(state) == PUZZLE_3X3.best_move


class TestActLargerBoards:
    def test_4x4_picks_best_move(self):
        state = state_with_moves(PUZZLE_4X4.moves, PUZZLE_4X4.n, PUZZLE_4X4.k)
        agent = MTDfIDAgent(16)
        assert agent.act(state) == PUZZLE_4X4.best_move

    def test_5x5_picks_best_move(self):
        state = state_with_moves(PUZZLE_5X5.moves, PUZZLE_5X5.n, PUZZLE_5X5.k)
        agent = MTDfIDAgent(25)
        assert agent.act(state) == PUZZLE_5X5.best_move


class TestActValidMove:
    def test_returns_in_bounds_move(self):
        state = fresh_state()
        agent = MTDfIDAgent(9)
        row, col = agent.act(state)
        assert 0 <= row < state.board.n
        assert 0 <= col < state.board.n

    def test_returns_empty_cell(self):
        state = fresh_state()
        agent = MTDfIDAgent(9)
        row, col = agent.act(state)
        assert state.board.is_empty(row, col)


class TestIterativeDeepening:
    """Tests specific to the iterative deepening structure of MTDfIDAgent."""

    def test_depth_zero_nonterminal_does_not_populate_tt(self):
        # depth=0 at a non-terminal root must leave TT empty (heuristic not stored).
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        score = agent._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, 0, tt)
        assert -1.0 <= score <= 1.0
        assert len(tt) == 0

    def test_tt_monotonically_grows_or_stays_across_depth_iterations(self):
        # TT entries must never be removed between depth iterations.
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        agent._mtdf(state, 0.0, 1, tt)
        size_after_d1 = len(tt)
        agent._mtdf(state, 0.3, 2, tt)
        size_after_d2 = len(tt)
        assert size_after_d2 >= size_after_d1

    def test_f_value_propagates_across_depth_iterations(self):
        # Simulate the act() ID loop manually and verify f stays at true value.
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        f = 0.0
        f = agent._mtdf(state, f, 1, tt)
        assert f == pytest.approx(1.0 - _EPS * 7)
        f = agent._mtdf(state, f, 2, tt)
        assert f == pytest.approx(1.0 - _EPS * 7)

    def test_best_move_stored_in_tt_after_shallow_search(self):
        # After depth=1, the TT must already have the winning best_move at root.
        # This verifies the ID loop seeds the TT with ordering hints for deeper depths.
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        agent._mtdf(state, 0.0, 1, tt)
        assert tt.best_move(state._hash) == (1, 2)

    def test_best_move_in_tt_preserved_after_deeper_iteration(self):
        # The root best_move written at depth=1 must survive a depth=2 iteration
        # on the same TT — the second pass may overwrite with the same or better move.
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        agent._mtdf(state, 0.0, 1, tt)
        agent._mtdf(state, 0.3, 2, tt)
        assert tt.best_move(state._hash) == (1, 2)

    def test_converged_f_enables_fast_deeper_iteration(self):
        # Passing f=0.3 (from depth=1) to _mtdf at depth=2 on a seeded TT must
        # return the same exact value — TT cutoffs fire immediately.
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        f = agent._mtdf(state, 0.0, 1, tt)
        f2 = agent._mtdf(state, f, 2, tt)
        assert f2 == pytest.approx(1.0 - _EPS * 7)

    def test_terminal_node_entry_persists_across_depth_iterations(self):
        # The winning terminal child (after X plays (1,2)) is stored in TT at depth=1.
        # Running depth=2 on the same TT must not evict it.
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = MTDfIDAgent(9)
        tt = TranspositionTable()
        agent._mtdf(state, 0.0, 1, tt)
        state.apply(1, 2)
        terminal_hash = state._hash
        state.undo()
        assert tt.lookup(terminal_hash) is not None  # stored after depth=1
        agent._mtdf(state, 0.3, 2, tt)
        assert tt.lookup(terminal_hash) is not None  # still in TT after depth=2


class TestAgreesWithNegamax:
    """MTDfIDAgent must choose identical moves to NegamaxAgent on well-defined positions."""

    def test_agrees_on_puzzle_3x3(self):
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        assert MTDfIDAgent(9).act(state) == NegamaxAgent(9).act(state)

    def test_agrees_on_empty_board(self):
        # MTDfIDAgent uses a heuristic at depth=0; on an empty board it prefers
        # the center (highest heuristic value) whereas NegamaxAgent without a
        # heuristic picks the first empty cell. Both moves are valid; we only
        # verify that MTDfIDAgent returns a legal cell.
        state = fresh_state()
        row, col = MTDfIDAgent(9).act(state)
        assert 0 <= row < state.board.n
        assert 0 <= col < state.board.n
        assert state.board.is_empty(row, col)

    def test_agrees_on_puzzle_4x4(self):
        state = state_with_moves(PUZZLE_4X4.moves, PUZZLE_4X4.n, PUZZLE_4X4.k)
        assert MTDfIDAgent(16).act(state) == NegamaxAgent(16).act(state)


class TestAgreesWithMTDf:
    """MTDfIDAgent must choose identical moves to MTDfAgent (same max_depth)."""

    def test_agrees_on_puzzle_3x3(self):
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        assert MTDfIDAgent(9).act(state) == MTDfAgent(9).act(state)

    def test_agrees_on_puzzle_4x4(self):
        state = state_with_moves(PUZZLE_4X4.moves, PUZZLE_4X4.n, PUZZLE_4X4.k)
        assert MTDfIDAgent(16).act(state) == MTDfAgent(16).act(state)
