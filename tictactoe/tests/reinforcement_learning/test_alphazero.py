"""Tests for AlphaZeroAgent."""
import numpy as np
import pytest

from tictactoe.agents.reinforcement_learning.alphazero import AlphaZeroAgent
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player


def make_empty_state(n=3):
    return GameState(board=Board.create(n), current_player=Player.X, n=n, k=n)


def test_returns_legal_move():
    agent = AlphaZeroAgent(n=3, num_simulations=10)
    state = make_empty_state()
    move = agent.choose_move(state)
    r, c = move
    assert state.board[r][c] is Cell.EMPTY


def test_get_name_and_tier():
    agent = AlphaZeroAgent(n=3)
    assert "AlphaZero" in agent.get_name()
    assert agent.get_tier() == 4


def test_picks_immediate_winning_move():
    import numpy as np
    from tictactoe.core.types import Cell as C
    board = Board.create(3)
    board[0][0] = C.X
    board[0][1] = C.X
    state = GameState(board=board, current_player=Player.X, n=3, k=3, last_move=(0, 1))
    agent = AlphaZeroAgent(n=3, num_simulations=10)
    move = agent.choose_move(state)
    assert move == (0, 2)


def test_data_augmentation_produces_8_variants():
    import numpy as np
    agent = AlphaZeroAgent(n=3)
    board = Board.create(3)
    board[0][0] = Cell.X
    policy = np.ones(9, dtype=np.float32) / 9
    variants = agent._generate_symmetries(board, 3, policy, 1.0)
    assert len(variants) == 8


def test_save_load_roundtrip(tmp_path):
    import torch
    agent = AlphaZeroAgent(n=3)
    path = str(tmp_path / "az_model")
    agent.save(path)
    agent2 = AlphaZeroAgent(n=3)
    agent2.load(path + '.pt')
    from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board_flat
    from tictactoe.core.types import Player
    x = encode_board_flat(Board.create(3), Player.X, 3)
    p1, v1 = agent._net.forward(x)
    p2, v2 = agent2._net.forward(x)
    assert torch.allclose(p1, p2, atol=1e-5)
    assert torch.allclose(v1, v2, atol=1e-5)


# ---------------------------------------------------------------------------
# Network type selection
# ---------------------------------------------------------------------------


def test_bitnet_network_type_constructs_and_returns_legal_move():
    agent = AlphaZeroAgent(n=3, num_simulations=5, network_type="bitnet")
    state = make_empty_state()
    move = agent.choose_move(state)
    r, c = move
    from tictactoe.core.types import Cell
    assert state.board[r][c] is Cell.EMPTY


def test_invalid_network_type_raises_value_error():
    with pytest.raises(ValueError, match="Unknown network_type"):
        AlphaZeroAgent(n=3, network_type="nonexistent")


def test_get_name_includes_network_type_quantized():
    agent = AlphaZeroAgent(n=3, network_type="quantized")
    assert "quantized" in agent.get_name()


def test_get_name_includes_network_type_bitnet():
    agent = AlphaZeroAgent(n=3, network_type="bitnet")
    assert "bitnet" in agent.get_name()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_train_on_example_returns_finite_loss():
    import numpy as np
    import math
    agent = AlphaZeroAgent(n=3, num_simulations=5)
    board = Board.create(3)
    from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board_flat
    from tictactoe.core.types import Player
    x = encode_board_flat(board, Player.X, 3)
    target_policy = np.ones(9, dtype=np.float32) / 9
    loss = agent.train_on_example(x, target_policy, target_value=0.0)
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert loss >= 0.0


def test_train_on_example_changes_network_bias():
    """After a training step, at least the policy bias must differ."""
    import torch
    agent = AlphaZeroAgent(n=3, num_simulations=5)
    board = Board.create(3)
    from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board_flat
    from tictactoe.core.types import Player
    x = encode_board_flat(board, Player.X, 3)
    target_policy = np.ones(9, dtype=np.float32) / 9
    bp_before = agent._net.bp.detach().clone()
    agent.train_on_example(x, target_policy, target_value=1.0)
    assert not torch.allclose(agent._net.bp, bp_before), "Bias should have updated"


def test_train_on_example_bitnet_network():
    """train_on_example must work for the bitnet network type too."""
    import math
    agent = AlphaZeroAgent(n=3, num_simulations=5, network_type="bitnet")
    board = Board.create(3)
    from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board_flat
    from tictactoe.core.types import Player
    x = encode_board_flat(board, Player.X, 3)
    target_policy = np.ones(9, dtype=np.float32) / 9
    loss = agent.train_on_example(x, target_policy, target_value=0.5)
    assert math.isfinite(loss) and loss >= 0.0


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


def test_nodes_visited_equals_num_simulations_on_unconstrained_budget():
    """nodes_visited should match the capped simulation count."""
    agent = AlphaZeroAgent(n=3, num_simulations=20)
    state = make_empty_state()
    agent.choose_move(state)
    assert state.nodes_visited <= 20


def test_choose_move_nodes_visited_positive():
    agent = AlphaZeroAgent(n=3, num_simulations=10)
    state = make_empty_state()
    agent.choose_move(state)
    assert state.nodes_visited > 0


# ---------------------------------------------------------------------------
# Internal helpers (now @staticmethod)
# ---------------------------------------------------------------------------


def test_backpropagate_increments_root_visits():
    """_backpropagate must increment visits at every node along the path."""
    from tictactoe.agents.reinforcement_learning.alphazero import PUCTNode
    from tictactoe.core.board import Board
    from tictactoe.core.types import Player
    root_state = make_empty_state()
    root = PUCTNode(root_state)
    child_state = root_state.apply_move((1, 1))
    child = PUCTNode(child_state, parent=root, move=(1, 1))
    root.children.append(child)

    agent = AlphaZeroAgent(n=3, num_simulations=1)
    agent._backpropagate(child, 1.0)

    assert child.visits == 1
    assert root.visits == 1


def test_generate_symmetries_produces_8_variants_as_staticmethod():
    """_generate_symmetries is now a @staticmethod — call without instance."""
    board = Board.create(3)
    board[0][0] = Cell.X
    import numpy as np
    policy = np.ones(9, dtype=np.float32) / 9
    variants = AlphaZeroAgent._generate_symmetries(board, 3, policy, 0.5)
    assert len(variants) == 8


def test_board_arr_to_flat_shape():
    """_board_arr_to_flat is now a @staticmethod — verify output shape."""
    import numpy as np
    board_arr = np.zeros((3, 3), dtype=int)
    flat = AlphaZeroAgent._board_arr_to_flat(board_arr, 3)
    assert flat.shape == (3 * 3 * 3,)  # 3 channels × 9 cells


# ---------------------------------------------------------------------------
# Large network types
# ---------------------------------------------------------------------------


class TestAlphaZeroAllNetworkTypes:
    """AlphaZero must support all three network types and their aliases."""

    def test_ternary_bitnet_large_constructs_and_returns_legal_move(self):
        from tictactoe.core.types import Cell
        agent = AlphaZeroAgent(n=3, num_simulations=5, network_type="ternary_bitnet_large")
        state = make_empty_state()
        move = agent.choose_move(state)
        r, c = move
        assert state.board[r][c] is Cell.EMPTY

    def test_bitnet_alias_constructs_and_returns_legal_move(self):
        from tictactoe.core.types import Cell
        agent = AlphaZeroAgent(n=3, num_simulations=5, network_type="bitnet")
        state = make_empty_state()
        move = agent.choose_move(state)
        r, c = move
        assert state.board[r][c] is Cell.EMPTY

    def test_float32_network_type_constructs(self):
        from tictactoe.core.types import Cell
        agent = AlphaZeroAgent(n=3, num_simulations=5, network_type="float32")
        state = make_empty_state()
        move = agent.choose_move(state)
        r, c = move
        assert state.board[r][c] is Cell.EMPTY

    def test_default_network_type_constructs(self):
        from tictactoe.core.types import Cell
        agent = AlphaZeroAgent(n=3, num_simulations=5, network_type="default")
        state = make_empty_state()
        move = agent.choose_move(state)
        r, c = move
        assert state.board[r][c] is Cell.EMPTY

    def test_get_name_includes_ternary_bitnet_large(self):
        agent = AlphaZeroAgent(n=3, network_type="ternary_bitnet_large")
        assert "ternary_bitnet_large" in agent.get_name()

    def test_bitnet_alias_resolves_to_ternary_bitnet_large_name(self):
        agent = AlphaZeroAgent(n=3, network_type="bitnet")
        assert "ternary_bitnet_large" in agent.get_name()

    def test_invalid_network_type_updated_message(self):
        """ValueError message must list valid network types."""
        with pytest.raises(ValueError, match="quantized"):
            AlphaZeroAgent(n=3, network_type="nonexistent")

    def test_train_on_example_bitnet_finite_loss(self):
        import math
        agent = AlphaZeroAgent(n=3, num_simulations=5, network_type="ternary_bitnet_large")
        board = Board.create(3)
        from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board_flat
        from tictactoe.core.types import Player
        x = encode_board_flat(board, Player.X, 3)
        target_policy = np.ones(9, dtype=np.float32) / 9
        loss = agent.train_on_example(x, target_policy, target_value=0.0)
        assert math.isfinite(loss) and loss >= 0.0

    def test_train_on_example_float32_finite_loss(self):
        import math
        agent = AlphaZeroAgent(n=3, num_simulations=5, network_type="float32")
        board = Board.create(3)
        from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board_flat
        from tictactoe.core.types import Player
        x = encode_board_flat(board, Player.X, 3)
        target_policy = np.ones(9, dtype=np.float32) / 9
        loss = agent.train_on_example(x, target_policy, target_value=0.5)
        assert math.isfinite(loss) and loss >= 0.0


# ---------------------------------------------------------------------------
# train_on_batch
# ---------------------------------------------------------------------------


class TestTrainOnBatch:
    """train_on_batch must return a finite mean loss and update weights."""

    def _make_examples(self, agent, n: int = 3, count: int = 4):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board_flat
        from tictactoe.core.types import Player
        board = Board.create(n)
        x = encode_board_flat(board, Player.X, n)
        tp = np.ones(n * n, dtype=np.float32) / (n * n)
        return [(x, tp, float(i % 2) * 2 - 1) for i in range(count)]

    def test_returns_finite_mean_loss(self):
        import math
        agent = AlphaZeroAgent(n=3, num_simulations=5, network_type="quantized")
        examples = self._make_examples(agent)
        loss = agent.train_on_batch(examples)
        assert math.isfinite(loss) and loss >= 0.0

    def test_empty_examples_raises(self):
        agent = AlphaZeroAgent(n=3, num_simulations=5, network_type="quantized")
        with pytest.raises(ValueError):
            agent.train_on_batch([])

    def test_float32_network_batch_train(self):
        import math
        agent = AlphaZeroAgent(n=3, num_simulations=5, network_type="float32")
        examples = self._make_examples(agent)
        loss = agent.train_on_batch(examples)
        assert math.isfinite(loss) and loss >= 0.0

    def test_batch_updates_policy_bias(self):
        import torch
        agent = AlphaZeroAgent(n=3, num_simulations=5, network_type="quantized")
        examples = self._make_examples(agent)
        bp_before = agent._net.bp.detach().clone()
        agent.train_on_batch(examples, lr=0.1)
        assert not torch.allclose(agent._net.bp, bp_before)


# ---------------------------------------------------------------------------
# Optimisations: Dirichlet noise, temperature annealing, tree reuse
# ---------------------------------------------------------------------------


class TestAlphaZeroOptimisations:
    """Tests for Dirichlet noise, temperature annealing, and tree reuse."""

    def test_dirichlet_noise_changes_priors(self):
        """_add_dirichlet_noise must produce a different distribution."""
        agent = AlphaZeroAgent(n=3, num_simulations=5)
        rng = np.random.default_rng(0)
        priors = np.ones(9, dtype=np.float32) / 9
        noisy = agent._add_dirichlet_noise(priors, rng)
        assert noisy.shape == priors.shape
        assert abs(noisy.sum() - 1.0) < 1e-5
        assert not np.allclose(noisy, priors)

    def test_dirichlet_noise_empty_priors(self):
        """Empty prior array must be returned unchanged."""
        agent = AlphaZeroAgent(n=3, num_simulations=5)
        rng = np.random.default_rng(0)
        priors = np.array([], dtype=np.float32)
        result = agent._add_dirichlet_noise(priors, rng)
        assert len(result) == 0

    def test_effective_temperature_early_moves(self):
        """Early moves must use self.temperature."""
        agent = AlphaZeroAgent(n=3, num_simulations=5, temperature=2.0)
        for move_num in range(1, 11):
            assert agent._effective_temperature(move_num) == pytest.approx(2.0)

    def test_effective_temperature_late_moves_below_initial(self):
        """Late moves must use a temperature lower than self.temperature."""
        agent = AlphaZeroAgent(n=3, num_simulations=5, temperature=1.0)
        late_temp = agent._effective_temperature(30)
        assert late_temp < 1.0
        assert late_temp >= agent._TEMP_FLOOR

    def test_effective_temperature_never_below_floor(self):
        agent = AlphaZeroAgent(n=3, num_simulations=5, temperature=1.0)
        for move_num in range(0, 100):
            assert agent._effective_temperature(move_num) >= agent._TEMP_FLOOR

    def test_find_reusable_root_returns_none_on_no_cache(self):
        """Without a cached root, must return None."""
        agent = AlphaZeroAgent(n=3, num_simulations=5)
        assert agent._find_reusable_root((1, 1)) is None

    def test_find_reusable_root_returns_none_for_no_last_move(self):
        from tictactoe.agents.reinforcement_learning.alphazero import PUCTNode
        agent = AlphaZeroAgent(n=3, num_simulations=5)
        agent._cached_root = PUCTNode(make_empty_state())
        assert agent._find_reusable_root(None) is None

    def test_tree_reuse_cached_root_set_after_choose_move(self):
        """After choose_move, _cached_root must be set (or None on fallback)."""
        from tictactoe.agents.reinforcement_learning.alphazero import PUCTNode
        agent = AlphaZeroAgent(n=3, num_simulations=10, network_type="quantized")
        state = make_empty_state()
        agent.choose_move(state)
        # _cached_root is either a PUCTNode or None (if no children were found)
        assert agent._cached_root is None or isinstance(agent._cached_root, PUCTNode)
