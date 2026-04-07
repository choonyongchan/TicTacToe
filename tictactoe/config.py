"""Configuration singleton for the TicTacToe AI framework.

Loads and validates config.toml once at the start of a run. All hyperparameters
for every algorithm family are consolidated here so that a single file controls
the entire experiment.

Dependency chain position: imports nothing from tictactoe.* — sits above all
other modules and is safely imported by _search_budget, agents, and main.
"""
from __future__ import annotations

import threading
import tomllib
import pathlib
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Raised when config.toml is missing, incomplete, or contains invalid values."""


# ---------------------------------------------------------------------------
# Dataclass hierarchy
#
# All fields are required — no defaults. Values must come from config.toml
# via load_config(). Constructing these dataclasses directly without providing
# every field will raise TypeError.
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    """Search algorithm hyperparameters."""
    id_max_depth: int
    aspiration_delta: float
    tt_size: int
    tss_max_depth: int
    mtdf_max_iterations: int


@dataclass
class BudgetConfig:
    """Per-move computation budget."""
    time_limit_ms: float
    node_budget: int
    fixed_depth: int


@dataclass
class MCTSConfig:
    """Monte Carlo Tree Search hyperparameters."""
    exploration_constant: float
    rollout_depth_limit: int
    rave_k: float
    rave_exploration_constant: float
    heuristic_rollout_depth: int
    alphazero_lite_simulations: int
    alphazero_lite_c_puct: float


@dataclass
class RLConfig:
    """Reinforcement learning hyperparameters."""
    dqn_buffer_capacity: int
    dqn_batch_size: int
    dqn_gamma: float
    dqn_lr: float
    dqn_epsilon: float
    dqn_target_update_freq: int
    ppo_epsilon_clip: float
    ppo_gamma: float
    ppo_lam: float
    ppo_lr: float
    tabq_alpha: float
    tabq_gamma: float
    alphazero_simulations: int
    alphazero_c_puct: float
    alphazero_temperature: float
    alphazero_lr: float


@dataclass
class GameConfig:
    """Game and benchmarking settings."""
    n: int
    k: int   # 0 means "use n"
    num_games: int
    seed: int


@dataclass
class AppConfig:
    """Top-level configuration container."""
    search: SearchConfig
    budget: BudgetConfig
    mcts: MCTSConfig
    rl: RLConfig
    game: GameConfig


# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_config: AppConfig | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str | pathlib.Path) -> None:
    """Load and validate config.toml, then populate the singleton.

    Must be called once from main() before any agents are instantiated.
    Raises ConfigError if:
    - The file does not exist.
    - The file has a TOML syntax error.
    - Any required section or key is absent.
    - Any value fails its bounds check.
    All validation errors are collected and reported together.

    Args:
        path: Path to config.toml.
    """
    global _config
    p = pathlib.Path(path)
    if not p.exists():
        raise ConfigError(
            f"Config file not found: {p}\n"
            "Create config.toml at the project root before running.\n"
            "A template with default values and documentation is provided in\n"
            "the repository root."
        )
    with p.open("rb") as f:
        try:
            raw = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Failed to parse {p}: {exc}") from exc

    cfg = _build_and_validate(raw)
    with _lock:
        _config = cfg


def get_config() -> AppConfig:
    """Return the active AppConfig.

    Raises ConfigError if load_config() was never called.

    Returns:
        The validated AppConfig populated by the most recent load_config() call.

    Raises:
        ConfigError: If load_config() was not called before this function.
    """
    if _config is None:
        raise ConfigError(
            "Config not loaded. Call load_config(path) before accessing get_config().\n"
            "If you are writing a unit test, construct AppConfig() directly instead."
        )
    return _config


# ---------------------------------------------------------------------------
# Internal builder + validator
# ---------------------------------------------------------------------------

def _build_and_validate(raw: dict) -> AppConfig:
    """Build AppConfig from a raw parsed TOML dict.

    Enforces that every key defined in each section dataclass is present in
    the corresponding TOML section. Unknown keys in the file are ignored
    (forward-compatibility for new keys added in future versions).
    Numeric values are checked against documented valid ranges.

    All problems are collected before raising, so users see every issue at once.

    Args:
        raw: Parsed TOML dict (output of tomllib.load).

    Returns:
        A fully validated AppConfig.

    Raises:
        ConfigError: If any required key is missing or any value is out of range.
    """
    errors: list[str] = []

    def _extract(cls, section_key: str) -> dict:
        """Pull all fields for one section dataclass; collect missing-key errors."""
        section_data = raw.get(section_key)
        if section_data is None:
            errors.append(f"[{section_key}]: section missing from config.toml")
            return {}
        result = {}
        for field_name in cls.__dataclass_fields__:
            if field_name not in section_data:
                errors.append(
                    f"{section_key}.{field_name}: required field missing"
                )
            else:
                result[field_name] = section_data[field_name]
        return result

    s = _extract(SearchConfig, "search")
    b = _extract(BudgetConfig, "budget")
    m = _extract(MCTSConfig, "mcts")
    r = _extract(RLConfig, "rl")
    g = _extract(GameConfig, "game")

    # Fail before bounds checks if any sections/keys are missing to avoid
    # spurious AttributeErrors on the extracted dicts.
    if errors:
        raise ConfigError(
            "Invalid config.toml — missing required fields:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    # Bounds validation — all violations collected before raising.
    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    def in_range(val, lo, hi, name: str,
                 inclusive_lo: bool = True, inclusive_hi: bool = True) -> None:
        lo_ok = (val >= lo) if inclusive_lo else (val > lo)
        hi_ok = (val <= hi) if inclusive_hi else (val < hi)
        lo_sym = "[" if inclusive_lo else "("
        hi_sym = "]" if inclusive_hi else ")"
        check(
            lo_ok and hi_ok,
            f"{name}: {val} is outside the valid range "
            f"{lo_sym}{lo}, {hi}{hi_sym}",
        )

    # [search]
    check(s["id_max_depth"] >= 1,
          "search.id_max_depth: must be >= 1")
    check(s["aspiration_delta"] > 0,
          "search.aspiration_delta: must be > 0")
    check(
        s["tt_size"] >= 1 and (s["tt_size"] & (s["tt_size"] - 1)) == 0,
        "search.tt_size: must be a positive power of 2 (e.g. 524288, 1048576, 2097152)",
    )
    check(s["tss_max_depth"] >= 1,
          "search.tss_max_depth: must be >= 1")
    check(s["mtdf_max_iterations"] >= 1,
          "search.mtdf_max_iterations: must be >= 1")

    # [budget]
    check(b["time_limit_ms"] > 0,
          "budget.time_limit_ms: must be > 0")
    check(b["node_budget"] >= 1,
          "budget.node_budget: must be >= 1")
    check(b["fixed_depth"] >= 1,
          "budget.fixed_depth: must be >= 1")

    # [mcts]
    check(m["exploration_constant"] >= 0,
          "mcts.exploration_constant: must be >= 0")
    check(m["rollout_depth_limit"] >= 1,
          "mcts.rollout_depth_limit: must be >= 1")
    check(m["rave_k"] > 0,
          "mcts.rave_k: must be > 0")
    check(m["rave_exploration_constant"] >= 0,
          "mcts.rave_exploration_constant: must be >= 0")
    check(m["heuristic_rollout_depth"] >= 1,
          "mcts.heuristic_rollout_depth: must be >= 1")
    check(m["alphazero_lite_simulations"] >= 1,
          "mcts.alphazero_lite_simulations: must be >= 1")
    check(m["alphazero_lite_c_puct"] >= 0,
          "mcts.alphazero_lite_c_puct: must be >= 0")

    # [rl]
    check(r["dqn_buffer_capacity"] >= 1,
          "rl.dqn_buffer_capacity: must be >= 1")
    check(r["dqn_batch_size"] >= 1,
          "rl.dqn_batch_size: must be >= 1")
    in_range(r["dqn_gamma"],  0, 1, "rl.dqn_gamma",  inclusive_lo=False)
    in_range(r["dqn_lr"],     0, 1, "rl.dqn_lr",     inclusive_lo=False)
    in_range(r["dqn_epsilon"], 0, 1, "rl.dqn_epsilon")
    check(r["dqn_target_update_freq"] >= 1,
          "rl.dqn_target_update_freq: must be >= 1")
    in_range(r["ppo_epsilon_clip"], 0, 1, "rl.ppo_epsilon_clip",
             inclusive_lo=False, inclusive_hi=False)
    in_range(r["ppo_gamma"], 0, 1, "rl.ppo_gamma",  inclusive_lo=False)
    in_range(r["ppo_lam"],   0, 1, "rl.ppo_lam",    inclusive_lo=False)
    in_range(r["ppo_lr"],    0, 1, "rl.ppo_lr",     inclusive_lo=False)
    in_range(r["tabq_alpha"], 0, 1, "rl.tabq_alpha", inclusive_lo=False)
    in_range(r["tabq_gamma"], 0, 1, "rl.tabq_gamma", inclusive_lo=False)
    check(r["alphazero_simulations"] >= 1,
          "rl.alphazero_simulations: must be >= 1")
    check(r["alphazero_c_puct"] >= 0,
          "rl.alphazero_c_puct: must be >= 0")
    check(r["alphazero_temperature"] > 0,
          "rl.alphazero_temperature: must be > 0")
    check(r["alphazero_lr"] > 0,
          "rl.alphazero_lr: must be > 0")

    # [game]
    check(g["n"] >= 3,
          "game.n: must be >= 3 (minimum board size for Tic-Tac-Toe)")
    check(g["k"] >= 0,
          "game.k: must be >= 0 (use 0 to default to n)")
    check(g["num_games"] >= 1,
          "game.num_games: must be >= 1")

    if errors:
        raise ConfigError(
            "Invalid config.toml — validation failed:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    return AppConfig(
        search=SearchConfig(**s),
        budget=BudgetConfig(**b),
        mcts=MCTSConfig(**m),
        rl=RLConfig(**r),
        game=GameConfig(**g),
    )
