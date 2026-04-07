"""Tests for the tictactoe.config module.

Covers:
- Unit tests: dataclass construction, load/get API, validation logic,
  bounds checking for every section, forward-compatibility with unknown keys,
  and multiple load calls.
- Integration test: load the real config.toml from the project root and
  verify all values are accessible and match expected defaults.
"""

from __future__ import annotations

import pathlib

import pytest

import tictactoe.config as _mod
from tictactoe.config import (
    AppConfig,
    BudgetConfig,
    ConfigError,
    GameConfig,
    MCTSConfig,
    RLConfig,
    SearchConfig,
    get_config,
    load_config,
)

# Path to the real config.toml in the project root
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
_REAL_CONFIG = _PROJECT_ROOT / "config.toml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_config():
    """Save and restore the config singleton around every test.

    This prevents one test's load_config() call from bleeding into others.
    """
    saved = _mod._config
    yield
    _mod._config = saved


def _write_toml(tmp_path: pathlib.Path, content: str) -> pathlib.Path:
    """Write *content* to a temp TOML file and return its path."""
    p = tmp_path / "config.toml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Minimal valid TOML (all required fields present and in bounds)
# ---------------------------------------------------------------------------

_VALID_TOML = """\
[search]
id_max_depth = 500
aspiration_delta = 25.0
tt_size = 524288
tss_max_depth = 8
mtdf_max_iterations = 30

[budget]
time_limit_ms = 500.0
node_budget = 50000
fixed_depth = 6

[mcts]
exploration_constant = 1.0
rollout_depth_limit = 100
rave_k = 300.0
rave_exploration_constant = 0.4
heuristic_rollout_depth = 40
alphazero_lite_simulations = 50
alphazero_lite_c_puct = 2.0

[rl]
dqn_buffer_capacity = 5000
dqn_batch_size = 16
dqn_gamma = 0.9
dqn_lr = 0.01
dqn_epsilon = 0.2
dqn_target_update_freq = 50
ppo_epsilon_clip = 0.1
ppo_gamma = 0.95
ppo_lam = 0.9
ppo_lr = 0.005
tabq_alpha = 0.2
tabq_gamma = 0.8
alphazero_simulations = 25
alphazero_c_puct = 2.0
alphazero_temperature = 0.5
alphazero_lr = 0.005

[game]
n = 5
k = 4
num_games = 50
seed = 7
"""


# ---------------------------------------------------------------------------
# Unit tests — AppConfig direct construction (no file)
# ---------------------------------------------------------------------------


class TestAppConfigDirectConstruction:
    """AppConfig sub-dataclasses require all fields — no silent defaults."""

    def test_search_config_requires_all_fields(self):
        with pytest.raises(TypeError):
            SearchConfig()

    def test_app_config_requires_all_fields(self):
        with pytest.raises(TypeError):
            AppConfig()


# ---------------------------------------------------------------------------
# Unit tests — get_config() before load_config()
# ---------------------------------------------------------------------------


class TestGetConfigUnloaded:
    """get_config() must raise when load_config() has not been called."""

    def test_raises_config_error(self):
        _mod._config = None
        with pytest.raises(ConfigError, match="not loaded"):
            get_config()

    def test_error_message_mentions_load_config(self):
        _mod._config = None
        with pytest.raises(ConfigError) as exc_info:
            get_config()
        assert "load_config" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Unit tests — load_config() file errors
# ---------------------------------------------------------------------------


class TestLoadConfigFileErrors:
    """load_config() fails fast with a clear message on file-level issues."""

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nonexistent.toml")

    def test_missing_file_error_contains_path(self, tmp_path):
        missing = tmp_path / "missing.toml"
        with pytest.raises(ConfigError) as exc_info:
            load_config(missing)
        assert str(missing) in str(exc_info.value)

    def test_invalid_toml_syntax_raises(self, tmp_path):
        p = _write_toml(tmp_path, "this is not valid toml ===")
        with pytest.raises(ConfigError, match="Failed to parse"):
            load_config(p)

    def test_accepts_path_as_string(self, tmp_path):
        p = _write_toml(tmp_path, _VALID_TOML)
        load_config(str(p))  # must accept str, not only Path
        assert get_config().search.id_max_depth == 500


# ---------------------------------------------------------------------------
# Unit tests — load_config() missing sections / keys
# ---------------------------------------------------------------------------


class TestLoadConfigMissingFields:
    """load_config() raises ConfigError listing every missing field."""

    def test_entirely_empty_file_reports_all_sections(self, tmp_path):
        p = _write_toml(tmp_path, "")
        with pytest.raises(ConfigError) as exc_info:
            load_config(p)
        msg = str(exc_info.value)
        for section in ("search", "budget", "mcts", "rl", "game"):
            assert section in msg

    def test_missing_one_section_reports_it(self, tmp_path):
        # Remove the [budget] section entirely
        toml = "\n".join(
            line for line in _VALID_TOML.splitlines()
            if not line.startswith("[budget]")
            and not any(
                line.startswith(k)
                for k in ("time_limit_ms", "node_budget", "fixed_depth")
            )
        )
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ConfigError) as exc_info:
            load_config(p)
        assert "budget" in str(exc_info.value)

    def test_missing_one_key_reports_it(self, tmp_path):
        # Remove id_max_depth from the [search] section
        toml = "\n".join(
            line for line in _VALID_TOML.splitlines()
            if not line.startswith("id_max_depth")
        )
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ConfigError) as exc_info:
            load_config(p)
        assert "id_max_depth" in str(exc_info.value)

    def test_multiple_missing_keys_all_reported(self, tmp_path):
        # Remove id_max_depth AND tss_max_depth — both must appear in the error
        toml = "\n".join(
            line for line in _VALID_TOML.splitlines()
            if not line.startswith("id_max_depth")
            and not line.startswith("tss_max_depth")
        )
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ConfigError) as exc_info:
            load_config(p)
        msg = str(exc_info.value)
        assert "id_max_depth" in msg
        assert "tss_max_depth" in msg


# ---------------------------------------------------------------------------
# Unit tests — bounds validation
# ---------------------------------------------------------------------------


def _toml_with(section: str, key: str, value) -> str:
    """Return a copy of _VALID_TOML with one field replaced."""
    lines = []
    for line in _VALID_TOML.splitlines():
        if line.startswith(f"{key} ="):
            lines.append(f"{key} = {value}")
        else:
            lines.append(line)
    return "\n".join(lines)


def _expect_bounds_error(tmp_path, section, key, bad_value, match=None):
    """Helper: write a TOML with one bad value and assert ConfigError is raised."""
    toml = _toml_with(section, key, bad_value)
    p = _write_toml(tmp_path, toml)
    pattern = match or key
    with pytest.raises(ConfigError, match=pattern):
        load_config(p)


class TestSearchBoundsValidation:
    def test_id_max_depth_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "search", "id_max_depth", 0)

    def test_id_max_depth_negative(self, tmp_path):
        _expect_bounds_error(tmp_path, "search", "id_max_depth", -5)

    def test_aspiration_delta_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "search", "aspiration_delta", 0.0)

    def test_aspiration_delta_negative(self, tmp_path):
        _expect_bounds_error(tmp_path, "search", "aspiration_delta", -1.0)

    def test_tt_size_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "search", "tt_size", 0)

    def test_tt_size_not_power_of_two(self, tmp_path):
        _expect_bounds_error(tmp_path, "search", "tt_size", 1000000)

    def test_tt_size_power_of_two_accepted(self, tmp_path):
        # 2^19 = 524288 is a valid power of two
        p = _write_toml(tmp_path, _toml_with("search", "tt_size", 524288))
        load_config(p)  # must not raise
        assert get_config().search.tt_size == 524288

    def test_tss_max_depth_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "search", "tss_max_depth", 0)

    def test_mtdf_max_iterations_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "search", "mtdf_max_iterations", 0)


class TestBudgetBoundsValidation:
    def test_time_limit_ms_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "budget", "time_limit_ms", 0.0)

    def test_time_limit_ms_negative(self, tmp_path):
        _expect_bounds_error(tmp_path, "budget", "time_limit_ms", -100.0)

    def test_node_budget_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "budget", "node_budget", 0)

    def test_fixed_depth_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "budget", "fixed_depth", 0)


class TestMCTSBoundsValidation:
    def test_exploration_constant_negative(self, tmp_path):
        _expect_bounds_error(tmp_path, "mcts", "exploration_constant", -0.1)

    def test_exploration_constant_zero_accepted(self, tmp_path):
        p = _write_toml(tmp_path, _toml_with("mcts", "exploration_constant", 0.0))
        load_config(p)  # 0 is a valid (pure greedy) constant

    def test_rollout_depth_limit_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "mcts", "rollout_depth_limit", 0)

    def test_rave_k_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "mcts", "rave_k", 0.0)

    def test_rave_k_negative(self, tmp_path):
        _expect_bounds_error(tmp_path, "mcts", "rave_k", -1.0)

    def test_heuristic_rollout_depth_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "mcts", "heuristic_rollout_depth", 0)

    def test_alphazero_lite_simulations_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "mcts", "alphazero_lite_simulations", 0)

    def test_alphazero_lite_c_puct_negative(self, tmp_path):
        _expect_bounds_error(tmp_path, "mcts", "alphazero_lite_c_puct", -0.5)


class TestRLBoundsValidation:
    def test_dqn_gamma_zero(self, tmp_path):
        # strictly > 0
        _expect_bounds_error(tmp_path, "rl", "dqn_gamma", 0.0)

    def test_dqn_gamma_above_one(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "dqn_gamma", 1.1)

    def test_dqn_gamma_one_accepted(self, tmp_path):
        p = _write_toml(tmp_path, _toml_with("rl", "dqn_gamma", 1.0))
        load_config(p)

    def test_dqn_lr_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "dqn_lr", 0.0)

    def test_dqn_lr_above_one_rejected(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "dqn_lr", 1.5)

    def test_dqn_lr_one_accepted(self, tmp_path):
        # lr = 1.0 is a valid (full gradient step) learning rate
        p = _write_toml(tmp_path, _toml_with("rl", "dqn_lr", 1.0))
        load_config(p)

    def test_dqn_epsilon_negative(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "dqn_epsilon", -0.1)

    def test_dqn_epsilon_zero_accepted(self, tmp_path):
        p = _write_toml(tmp_path, _toml_with("rl", "dqn_epsilon", 0.0))
        load_config(p)

    def test_dqn_epsilon_one_accepted(self, tmp_path):
        p = _write_toml(tmp_path, _toml_with("rl", "dqn_epsilon", 1.0))
        load_config(p)

    def test_dqn_epsilon_above_one(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "dqn_epsilon", 1.01)

    def test_ppo_epsilon_clip_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "ppo_epsilon_clip", 0.0)

    def test_ppo_epsilon_clip_one(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "ppo_epsilon_clip", 1.0)

    def test_ppo_gamma_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "ppo_gamma", 0.0)

    def test_ppo_lam_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "ppo_lam", 0.0)

    def test_ppo_lr_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "ppo_lr", 0.0)

    def test_tabq_alpha_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "tabq_alpha", 0.0)

    def test_tabq_gamma_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "tabq_gamma", 0.0)

    def test_alphazero_simulations_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "alphazero_simulations", 0)

    def test_alphazero_c_puct_negative(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "alphazero_c_puct", -0.5)

    def test_alphazero_temperature_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "alphazero_temperature", 0.0)

    def test_alphazero_lr_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "alphazero_lr", 0.0)

    def test_dqn_buffer_capacity_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "dqn_buffer_capacity", 0)

    def test_dqn_batch_size_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "dqn_batch_size", 0)

    def test_dqn_target_update_freq_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "rl", "dqn_target_update_freq", 0)


class TestGameBoundsValidation:
    def test_n_below_three(self, tmp_path):
        _expect_bounds_error(tmp_path, "game", "n", 2)

    def test_n_one(self, tmp_path):
        _expect_bounds_error(tmp_path, "game", "n", 1)

    def test_k_negative(self, tmp_path):
        _expect_bounds_error(tmp_path, "game", "k", -1)

    def test_k_zero_accepted(self, tmp_path):
        p = _write_toml(tmp_path, _toml_with("game", "k", 0))
        load_config(p)  # 0 means "use n"

    def test_num_games_zero(self, tmp_path):
        _expect_bounds_error(tmp_path, "game", "num_games", 0)


# ---------------------------------------------------------------------------
# Unit tests — successful load and value access
# ---------------------------------------------------------------------------


class TestSuccessfulLoad:
    """After a valid load, get_config() returns the correct values."""

    def test_loaded_values_match_toml(self, tmp_path):
        p = _write_toml(tmp_path, _VALID_TOML)
        load_config(p)
        cfg = get_config()

        assert cfg.search.id_max_depth == 500
        assert cfg.search.aspiration_delta == 25.0
        assert cfg.search.tt_size == 524288
        assert cfg.search.tss_max_depth == 8
        assert cfg.search.mtdf_max_iterations == 30

        assert cfg.budget.time_limit_ms == 500.0
        assert cfg.budget.node_budget == 50000
        assert cfg.budget.fixed_depth == 6

        assert cfg.mcts.exploration_constant == 1.0
        assert cfg.mcts.rollout_depth_limit == 100
        assert cfg.mcts.rave_k == 300.0
        assert cfg.mcts.rave_exploration_constant == 0.4
        assert cfg.mcts.heuristic_rollout_depth == 40
        assert cfg.mcts.alphazero_lite_simulations == 50
        assert cfg.mcts.alphazero_lite_c_puct == 2.0

        assert cfg.rl.dqn_buffer_capacity == 5000
        assert cfg.rl.dqn_batch_size == 16
        assert cfg.rl.dqn_gamma == pytest.approx(0.9)
        assert cfg.rl.dqn_lr == pytest.approx(0.01)
        assert cfg.rl.dqn_epsilon == pytest.approx(0.2)
        assert cfg.rl.dqn_target_update_freq == 50
        assert cfg.rl.ppo_epsilon_clip == pytest.approx(0.1)
        assert cfg.rl.ppo_gamma == pytest.approx(0.95)
        assert cfg.rl.ppo_lam == pytest.approx(0.9)
        assert cfg.rl.ppo_lr == pytest.approx(0.005)
        assert cfg.rl.tabq_alpha == pytest.approx(0.2)
        assert cfg.rl.tabq_gamma == pytest.approx(0.8)
        assert cfg.rl.alphazero_simulations == 25
        assert cfg.rl.alphazero_c_puct == pytest.approx(2.0)
        assert cfg.rl.alphazero_temperature == pytest.approx(0.5)
        assert cfg.rl.alphazero_lr == pytest.approx(0.005)

        assert cfg.game.n == 5
        assert cfg.game.k == 4
        assert cfg.game.num_games == 50
        assert cfg.game.seed == 7

    def test_unknown_keys_ignored(self, tmp_path):
        """Extra keys in the TOML file are silently ignored (forward-compat)."""
        toml = _VALID_TOML + "\n[search]\nsome_future_key = 99\n"
        # Re-append the whole [search] section with the extra key
        toml = _VALID_TOML + "\nsome_unknown_top_level = true\n"
        p = _write_toml(tmp_path, toml)
        load_config(p)  # must not raise

    def test_subsequent_load_replaces_config(self, tmp_path):
        """Calling load_config() twice replaces the singleton."""
        p1 = _write_toml(tmp_path / "c1.toml" if False else tmp_path, _VALID_TOML)
        load_config(p1)
        assert get_config().search.id_max_depth == 500

        toml2 = _toml_with("search", "id_max_depth", 999)
        p2 = tmp_path / "config2.toml"
        p2.write_text(toml2, encoding="utf-8")
        load_config(p2)
        assert get_config().search.id_max_depth == 999

    def test_get_config_returns_same_object(self, tmp_path):
        """get_config() must return the same AppConfig instance on repeated calls."""
        p = _write_toml(tmp_path, _VALID_TOML)
        load_config(p)
        assert get_config() is get_config()

    def test_multiple_bounds_errors_all_reported(self, tmp_path):
        """All bounds violations are reported in a single ConfigError."""
        toml = (
            _VALID_TOML
            .replace("id_max_depth = 500", "id_max_depth = 0")
            .replace("aspiration_delta = 25.0", "aspiration_delta = -1.0")
            .replace("node_budget = 50000", "node_budget = 0")
        )
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ConfigError) as exc_info:
            load_config(p)
        msg = str(exc_info.value)
        assert "id_max_depth" in msg
        assert "aspiration_delta" in msg
        assert "node_budget" in msg


# ---------------------------------------------------------------------------
# Unit tests — error message quality
# ---------------------------------------------------------------------------


class TestErrorMessages:
    """Error messages must be specific enough to act on without reading source."""

    def test_missing_file_message_actionable(self, tmp_path):
        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path / "ghost.toml")
        msg = str(exc_info.value)
        assert "config.toml" in msg.lower() or "config file" in msg.lower()

    def test_missing_key_names_field(self, tmp_path):
        toml = "\n".join(
            l for l in _VALID_TOML.splitlines()
            if not l.startswith("dqn_gamma")
        )
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ConfigError) as exc_info:
            load_config(p)
        assert "dqn_gamma" in str(exc_info.value)

    def test_bounds_error_includes_value(self, tmp_path):
        p = _write_toml(tmp_path, _toml_with("search", "id_max_depth", -42))
        with pytest.raises(ConfigError) as exc_info:
            load_config(p)
        # The offending value or field name should appear in the message
        msg = str(exc_info.value)
        assert "id_max_depth" in msg

    def test_get_config_unloaded_message_mentions_load_config(self):
        _mod._config = None
        with pytest.raises(ConfigError) as exc_info:
            get_config()
        assert "load_config" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Integration test — real config.toml
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _REAL_CONFIG.exists(),
    reason="config.toml not found in project root — skipping integration test",
)
class TestRealConfigToml:
    """Load the actual config.toml shipped with the project and verify it."""

    def test_loads_without_error(self):
        load_config(_REAL_CONFIG)  # must not raise

    def test_default_search_values(self):
        load_config(_REAL_CONFIG)
        cfg = get_config()
        assert cfg.search.id_max_depth == 1000
        assert cfg.search.aspiration_delta == 50.0
        assert cfg.search.tt_size == 1048576
        assert cfg.search.tss_max_depth == 10
        assert cfg.search.mtdf_max_iterations == 50

    def test_default_budget_values(self):
        load_config(_REAL_CONFIG)
        cfg = get_config()
        assert cfg.budget.time_limit_ms == 1000.0
        assert cfg.budget.node_budget == 100000
        assert cfg.budget.fixed_depth == 4

    def test_default_mcts_values(self):
        load_config(_REAL_CONFIG)
        cfg = get_config()
        assert cfg.mcts.exploration_constant == pytest.approx(1.414)
        assert cfg.mcts.rollout_depth_limit == 200
        assert cfg.mcts.rave_k == 500.0
        assert cfg.mcts.rave_exploration_constant == 0.5
        assert cfg.mcts.heuristic_rollout_depth == 50
        assert cfg.mcts.alphazero_lite_simulations == 100
        assert cfg.mcts.alphazero_lite_c_puct == 1.0

    def test_default_rl_values(self):
        load_config(_REAL_CONFIG)
        cfg = get_config()
        assert cfg.rl.dqn_buffer_capacity == 10000
        assert cfg.rl.dqn_batch_size == 32
        assert cfg.rl.dqn_gamma == pytest.approx(0.95)
        assert cfg.rl.dqn_lr == pytest.approx(0.001)
        assert cfg.rl.dqn_epsilon == pytest.approx(0.1)
        assert cfg.rl.dqn_target_update_freq == 100
        assert cfg.rl.ppo_epsilon_clip == pytest.approx(0.2)
        assert cfg.rl.ppo_gamma == pytest.approx(0.99)
        assert cfg.rl.ppo_lam == pytest.approx(0.95)
        assert cfg.rl.ppo_lr == pytest.approx(0.001)
        assert cfg.rl.tabq_alpha == pytest.approx(0.1)
        assert cfg.rl.tabq_gamma == pytest.approx(0.9)
        assert cfg.rl.alphazero_simulations == 50
        assert cfg.rl.alphazero_c_puct == pytest.approx(1.0)
        assert cfg.rl.alphazero_temperature == pytest.approx(1.0)
        assert cfg.rl.alphazero_lr == pytest.approx(0.001)

    def test_default_game_values(self):
        load_config(_REAL_CONFIG)
        cfg = get_config()
        assert cfg.game.n == 3
        assert cfg.game.k == 0  # 0 means "use n"
        assert cfg.game.num_games == 100
        assert cfg.game.seed == 42

    def test_all_bounds_are_valid(self):
        """Smoke test: the shipped config.toml passes every bounds check."""
        load_config(_REAL_CONFIG)  # would raise ConfigError if any bound fails

    def test_path_given_as_string(self):
        load_config(str(_REAL_CONFIG))
        assert get_config().search.id_max_depth == 1000
