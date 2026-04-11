"""Agent registry — maps UI agent IDs to constructors and display metadata.

Agents are imported lazily so the web module can start without fully loading
every algorithm.  All agents accept match_config=None (they default to a
1 s time-controlled budget when no config is provided).
"""
from __future__ import annotations

from typing import Any

from tictactoe.benchmark.metrics import MatchConfig

# ---------------------------------------------------------------------------
# Metadata exposed to the frontend
# ---------------------------------------------------------------------------

AGENTS: list[dict[str, Any]] = [
    {
        "id": "random",
        "name": "Random",
        "tier": 0,
        "description": "Uniform-random legal move. Fast baseline.",
    },
    {
        "id": "minimax_ab",
        "name": "Minimax α-β",
        "tier": 1,
        "description": "Exact minimax search with alpha-beta pruning.",
    },
    {
        "id": "negamax",
        "name": "Negamax",
        "tier": 1,
        "description": "Negamax formulation of minimax.",
    },
    {
        "id": "negascout",
        "name": "Negascout",
        "tier": 1,
        "description": "Principal variation search — faster than α-β near the PV.",
    },
    {
        "id": "mtdf",
        "name": "MTD(f)",
        "tier": 1,
        "description": "Memory-enhanced test driver with iterative narrowing.",
    },
    {
        "id": "minimax_ab_enhanced",
        "name": "Minimax α-β+",
        "tier": 2,
        "description": "Minimax with IDDFS, TSS, transposition table, killer & history heuristics.",
    },
    {
        "id": "negamax_enhanced",
        "name": "Negamax+",
        "tier": 2,
        "description": "Negamax with iterative deepening and all heuristic enhancements.",
    },
    {
        "id": "negascout_enhanced",
        "name": "Negascout+",
        "tier": 2,
        "description": "Negascout with iterative deepening and all heuristic enhancements.",
    },
    {
        "id": "mtdf_enhanced",
        "name": "MTD(f)+",
        "tier": 2,
        "description": "MTD(f) with iterative deepening and full heuristic suite.",
    },
    {
        "id": "mcts_vanilla",
        "name": "MCTS",
        "tier": 3,
        "description": "Vanilla Monte Carlo Tree Search with UCT selection.",
    },
    {
        "id": "mcts_rave",
        "name": "MCTS RAVE",
        "tier": 3,
        "description": "MCTS with AMAF / RAVE move-value sharing across the tree.",
    },
    {
        "id": "mcts_heuristic",
        "name": "MCTS Heuristic",
        "tier": 3,
        "description": "MCTS with heuristic-guided (non-random) rollouts.",
    },
    {
        "id": "mcts_alphazero",
        "name": "AlphaZero",
        "tier": 4,
        "description": "PUCT MCTS guided by a PolicyValueNetwork (policy + value heads).",
    },
]

_AGENT_IDS = {a["id"] for a in AGENTS}


def make_match_config(
    match_mode: str,
    time_limit_ms: float,
    node_budget: int,
    fixed_depth: int,
) -> MatchConfig:
    """Build a MatchConfig from the settings supplied by the frontend."""
    if match_mode == "node":
        return MatchConfig.node_controlled(node_budget)
    if match_mode == "depth":
        return MatchConfig.depth_controlled(fixed_depth)
    return MatchConfig.time_controlled(time_limit_ms)


def create_agent(agent_id: str, match_config: MatchConfig):
    """Instantiate the named agent. Raises ValueError for unknown ids."""
    if agent_id not in _AGENT_IDS:
        raise ValueError(f"Unknown agent id: {agent_id!r}")

    mc = match_config  # shorthand

    if agent_id == "random":
        from tictactoe.agents.random_agent import RandomAgent
        return RandomAgent()

    if agent_id == "minimax_ab":
        from tictactoe.agents.classic_search.minimax_ab import MinimaxAB
        return MinimaxAB(match_config=mc)

    if agent_id == "negamax":
        from tictactoe.agents.classic_search.negamax import NegaMax
        return NegaMax(match_config=mc)

    if agent_id == "negascout":
        from tictactoe.agents.classic_search.negascout import NegaScout
        return NegaScout(match_config=mc)

    if agent_id == "mtdf":
        from tictactoe.agents.classic_search.mtdf import MTDf
        return MTDf(match_config=mc)

    if agent_id == "minimax_ab_enhanced":
        from tictactoe.agents.heuristic_search.minimax_ab_enhanced import MinimaxABEnhanced
        return MinimaxABEnhanced(match_config=mc)

    if agent_id == "negamax_enhanced":
        from tictactoe.agents.heuristic_search.negamax_enhanced import NegaMaxEnhanced
        return NegaMaxEnhanced(match_config=mc)

    if agent_id == "negascout_enhanced":
        from tictactoe.agents.heuristic_search.negascout_enhanced import NegaScoutEnhanced
        return NegaScoutEnhanced(match_config=mc)

    if agent_id == "mtdf_enhanced":
        from tictactoe.agents.heuristic_search.mtdf_enhanced import MTDfEnhanced
        return MTDfEnhanced(match_config=mc)

    if agent_id == "mcts_vanilla":
        from tictactoe.agents.monte_carlo.mcts_vanilla import MCTSVanilla
        return MCTSVanilla(match_config=mc)

    if agent_id == "mcts_rave":
        from tictactoe.agents.monte_carlo.mcts_rave import MCTSRave
        return MCTSRave(match_config=mc)

    if agent_id == "mcts_heuristic":
        from tictactoe.agents.monte_carlo.mcts_heuristic_rollout import MCTSHeuristicRollout
        return MCTSHeuristicRollout(match_config=mc)

    if agent_id == "mcts_alphazero":
        from tictactoe.agents.reinforcement_learning.alphazero import AlphaZeroAgent
        return AlphaZeroAgent(match_config=mc)

    raise ValueError(f"Agent {agent_id!r} has no constructor mapping")
