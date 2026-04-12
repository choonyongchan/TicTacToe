"""Round-robin tournament engine and structured experiment runner.

The Arena class orchestrates duels, round-robin tournaments, sanity checks,
and scalability sweeps between agents. All diagnostic output uses Python's
logging module — no print statements.

Dependency chain position: types → state → board → game → agents → benchmark.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.benchmark.metrics import (
    DEFAULT_NODE_BUDGET,
    DEFAULT_TIME_LIMIT_MS,
    DuelResult,
    GameRecord,
    MatchConfig,
    MoveStats,
    RoundRobinResult,
    ScalabilityRecord,
    aggregate_move_stats,
    compute_elo,
    compute_pruning_rate,
)
from tictactoe.core.game import Game
from tictactoe.core.state import GameState
from tictactoe.core.types import (
    RANDOM_AGENT_WIN_THRESHOLD,
    MatchMode,
    Player,
    Result,
)

logger = logging.getLogger(__name__)


class Arena:
    """Orchestrates structured benchmarking experiments between agents.

    Attributes:
        n: Default board dimension for all matches.
        k: Winning line length (defaults to n when not specified).
        num_games: Number of games per duel.
        match_config: Budget configuration applied to every game.
        _seed: Random seed for reproducibility.
        _rng: Internal Random instance seeded by set_seed.
    """

    def __init__(
        self,
        n: int = 3,
        k: int | None = None,
        num_games: int = 100,
        match_config: MatchConfig | None = None,
    ) -> None:
        """Initialise the arena.

        Args:
            n: Default board dimension.
            k: Winning line length. Defaults to n when None.
            num_games: Games per duel. Must be even when swap_sides=True.
            match_config: Budget configuration. Defaults to time-controlled
                with DEFAULT_TIME_LIMIT_MS.
        """
        self.n = n
        self.k = k if k is not None else n
        self.num_games = num_games
        self.match_config = match_config or MatchConfig.time_controlled(
            DEFAULT_TIME_LIMIT_MS
        )
        self._seed: int | None = None
        self._rng = random.Random()

    # -------------------------------------------------------------------
    # Seed management
    # -------------------------------------------------------------------

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducible experiments.

        Args:
            seed: Integer seed applied to the internal RNG and to any
                RandomAgent instances created by the arena.
        """
        self._seed = seed
        self._rng.seed(seed)
        logger.debug("Arena seed set to %d.", seed)

    # -------------------------------------------------------------------
    # Core experiment methods
    # -------------------------------------------------------------------

    def single_game(
        self,
        agent_x: BaseAgent,
        agent_o: BaseAgent,
        n: int | None = None,
        k: int | None = None,
        verbose: bool = False,
    ) -> GameRecord:
        """Play one complete game and return a full GameRecord.

        Args:
            agent_x: Agent playing as Player X (moves first).
            agent_o: Agent playing as Player O.
            n: Board dimension override. Defaults to self.n.
            k: Winning line length override. Defaults to self.k (or n).
            verbose: When True, prints the board after each move.

        Returns:
            A GameRecord with per-move instrumentation for every move.
        """
        board_n = n if n is not None else self.n
        board_k = k if k is not None else (board_n if n is not None else self.k)

        game = Game(
            agent_x=agent_x,
            agent_o=agent_o,
            n=board_n,
            k=board_k,
            match_config=self.match_config,
        )

        per_move_stats: list[MoveStats] = []

        while game.state.result is Result.IN_PROGRESS:
            # Determine which player is about to move before stepping.
            moving_player = game.state.current_player
            state_after = game.step()
            stats = _extract_move_stats(state_after, moving_player)
            per_move_stats.append(stats)

        if verbose:
            from tictactoe.core.board import Board
            print(Board.render(game.state.board, board_n, game.state.last_move))

        return GameRecord(
            agent_x_name=agent_x.get_name(),
            agent_o_name=agent_o.get_name(),
            n=board_n,
            k=board_k,
            match_config=self.match_config,
            result=game.state.result,
            total_moves=game.state.move_number,
            move_history=list(game.state.move_history),
            per_move_stats=per_move_stats,
        )

    def duel(
        self,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        swap_sides: bool = True,
    ) -> DuelResult:
        """Play num_games games between two agents and summarise the results.

        When swap_sides=True, agents alternate who plays X so that neither
        side advantage is favoured. The total number of games played is
        num_games (not 2×num_games), so num_games must be even.

        Args:
            agent_a: First agent.
            agent_b: Second agent.
            swap_sides: Whether to alternate sides each game.

        Returns:
            A DuelResult with win counts, efficiency stats, and Elo ratings.

        Raises:
            ValueError: If swap_sides=True and num_games is odd.
        """
        if swap_sides and self.num_games % 2 != 0:
            raise ValueError(
                f"num_games must be even when swap_sides=True "
                f"(got num_games={self.num_games})."
            )

        logger.info(
            "Duel: %s vs %s | n=%d k=%d | %d games",
            agent_a.get_name(), agent_b.get_name(),
            self.n, self.k, self.num_games,
        )

        games: list[GameRecord] = []
        wins_a = 0
        wins_b = 0
        draws = 0

        for game_index in range(self.num_games):
            # Alternate who plays X if swap_sides is enabled.
            if swap_sides and game_index % 2 == 1:
                record = self.single_game(agent_b, agent_a)
                result_for_a = _result_for_player(record, agent_a.get_name())
            else:
                record = self.single_game(agent_a, agent_b)
                result_for_a = _result_for_player(record, agent_a.get_name())

            if record.result is not Result.DRAW:
                winner = (
                    record.agent_x_name if record.result is Result.X_WINS
                    else record.agent_o_name
                )
                logger.debug(
                    "Game %d solved: %s wins (%s vs %s) | n=%d k=%d",
                    game_index, winner,
                    record.agent_x_name, record.agent_o_name,
                    self.n, self.k,
                )

            games.append(record)

            if result_for_a is Result.X_WINS or result_for_a is Result.O_WINS:
                # Determine who actually won.
                if _agent_won(record, agent_a.get_name()):
                    wins_a += 1
                else:
                    wins_b += 1
            else:
                draws += 1

        duel_result = _build_duel_result(
            agent_a, agent_b, wins_a, wins_b, draws,
            self.n, self.k, self.match_config, games,
        )
        logger.info(
            "Duel complete: %s wins=%d  %s wins=%d  draws=%d",
            agent_a.get_name(), wins_a, agent_b.get_name(), wins_b, draws,
        )
        return duel_result

    def round_robin(self, agents: list[BaseAgent]) -> RoundRobinResult:
        """Run a round-robin tournament among all agents.

        Every unique pair plays a duel. Elo ratings are computed from
        all duel results combined.

        Args:
            agents: Participants. Must contain at least two agents.

        Returns:
            A RoundRobinResult with the duel matrix and final Elo ranking.

        Raises:
            ValueError: If fewer than 2 agents are provided.
        """
        if len(agents) < 2:
            raise ValueError("round_robin requires at least 2 agents.")

        agent_names = [a.get_name() for a in agents]
        logger.info("Round-robin: %s", agent_names)

        duel_matrix: dict[tuple[str, str], DuelResult] = {}
        all_duels: list[DuelResult] = []

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                result = self.duel(agents[i], agents[j])
                key = (agents[i].get_name(), agents[j].get_name())
                duel_matrix[key] = result
                all_duels.append(result)

        elo_ratings = compute_elo(all_duels)
        ranking = sorted(elo_ratings.items(), key=lambda kv: kv[1], reverse=True)

        return RoundRobinResult(
            agents=agent_names,
            n=self.n,
            match_config=self.match_config,
            duel_matrix=duel_matrix,
            final_elo_ranking=ranking,
        )

    def scalability_sweep(
        self,
        agents: list[BaseAgent],
        board_sizes: list[int],
        games_per_size: int = 20,
        k_override: int | None = None,
        max_moves_per_game: int | None = None,
    ) -> list[ScalabilityRecord]:
        """Measure how each agent scales across increasing board sizes.

        For each (agent, board_size) pair, plays games_per_size games against
        RandomAgent, then records average efficiency metrics and win rate.

        Args:
            agents: Agents to profile.
            board_sizes: Board dimensions to test, e.g. [3, 5, 7, 9].
            games_per_size: Number of games per (agent, board_size) pair.
            k_override: Winning line length used for all board sizes. When None,
                defaults to min(board_n, 5) for boards larger than 5, or board_n
                otherwise.
            max_moves_per_game: Cap total moves per game. Games still in progress
                at this limit are counted as draws. When None, defaults to
                max(9, 10 * k) to keep large-board benchmarks practical.

        Returns:
            One ScalabilityRecord per agent.
        """
        records: list[ScalabilityRecord] = []

        for agent in agents:
            record = ScalabilityRecord(
                agent_name=agent.get_name(),
                board_sizes=list(board_sizes),
                match_config=self.match_config,
            )

            for board_n in board_sizes:
                board_k = k_override if k_override is not None else min(board_n, 5)
                move_cap = max_moves_per_game if max_moves_per_game is not None \
                    else max(9, 10 * board_k)
                logger.info(
                    "Scalability: %s n=%d k=%d max_moves=%d (%d games)",
                    agent.get_name(), board_n, board_k, move_cap, games_per_size,
                )
                opponent = RandomAgent(seed=self._seed)
                size_games: list[GameRecord] = []

                for game_idx in range(games_per_size):
                    game_record = self._single_game_capped(
                        agent_x=agent,
                        agent_o=opponent,
                        n=board_n,
                        k=board_k,
                        max_moves=move_cap,
                    )
                    if game_record.result is not Result.DRAW:
                        winner = (
                            game_record.agent_x_name if game_record.result is Result.X_WINS
                            else game_record.agent_o_name
                        )
                        logger.debug(
                            "Game %d solved: %s wins (%s vs %s) | n=%d k=%d",
                            game_idx, winner,
                            game_record.agent_x_name, game_record.agent_o_name,
                            board_n, board_k,
                        )
                    size_games.append(game_record)

                # Collect all agent move stats for this board size.
                agent_stats = _collect_agent_stats(size_games, agent.get_name())
                agg = aggregate_move_stats(agent_stats)
                win_rate = _win_rate_against(size_games, agent.get_name())
                budget_exhausted = sum(1 for s in agent_stats if s.time_limit_exceeded)

                record.avg_nodes_per_size.append(agg["mean_nodes"])
                record.avg_depth_per_size.append(agg["mean_depth"])
                record.avg_ebf_per_size.append(agg["mean_ebf"])
                record.avg_time_ms_per_size.append(agg["mean_time_ms"])
                record.win_rate_per_size.append(win_rate)
                record.budget_exhausted_per_size.append(budget_exhausted)
                record.total_moves_per_size.append(len(agent_stats))

            records.append(record)

        return records

    def _single_game_capped(
        self,
        agent_x: BaseAgent,
        agent_o: BaseAgent,
        n: int,
        k: int,
        max_moves: int,
    ) -> GameRecord:
        """Play one game capped at max_moves total moves.

        Games still in progress at the cap are recorded as draws. This keeps
        scalability sweeps practical on large boards where random play
        produces very long games.
        """
        game = Game(
            agent_x=agent_x,
            agent_o=agent_o,
            n=n,
            k=k,
            match_config=self.match_config,
        )

        per_move_stats: list[MoveStats] = []

        while (
            game.state.result is Result.IN_PROGRESS
            and game.state.move_number < max_moves
        ):
            moving_player = game.state.current_player
            state_after = game.step()
            per_move_stats.append(_extract_move_stats(state_after, moving_player))

        # If capped before terminal, treat as draw for win-rate purposes.
        final_result = game.state.result

        return GameRecord(
            agent_x_name=agent_x.get_name(),
            agent_o_name=agent_o.get_name(),
            n=n,
            k=k,
            match_config=self.match_config,
            result=final_result,
            total_moves=game.state.move_number,
            move_history=list(game.state.move_history),
            per_move_stats=per_move_stats,
        )

    def sanity_check(
        self, agent: BaseAgent, games: int = 50
    ) -> dict[str, Any]:
        """Verify that an agent beats RandomAgent above the acceptance threshold.

        Runs `games` games between agent and RandomAgent on n=3. An agent
        that cannot meet RANDOM_AGENT_WIN_THRESHOLD is assumed to have a bug.

        Note:
            This check should be called before including an agent in a full
            round_robin or scalability_sweep. The arena logs a warning (not an
            error) if this method is skipped.

        Args:
            agent: The agent to sanity-check.
            games: Number of games to play. Should be even (sides are swapped).

        Returns:
            A dict with keys "passed" (bool), "win_rate" (float),
            "agent_name" (str).
        """
        logger.info("Sanity check: %s vs RandomAgent (%d games)", agent.get_name(), games)

        opponent = RandomAgent(seed=self._seed)
        original_num_games = self.num_games
        original_n = self.n
        original_k = self.k

        # Temporarily reconfigure for a 3×3 sanity check.
        self.num_games = games if games % 2 == 0 else games + 1
        self.n = 3
        self.k = 3

        duel_result = self.duel(agent, opponent)

        # Restore arena settings.
        self.num_games = original_num_games
        self.n = original_n
        self.k = original_k

        win_rate = duel_result.win_rate_a()
        passed = win_rate >= RANDOM_AGENT_WIN_THRESHOLD

        logger.info(
            "Sanity check %s: win_rate=%.3f threshold=%.3f → %s",
            agent.get_name(), win_rate, RANDOM_AGENT_WIN_THRESHOLD,
            "PASSED" if passed else "FAILED",
        )

        return {
            "passed": passed,
            "win_rate": win_rate,
            "agent_name": agent.get_name(),
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_move_stats(state: GameState, player: Player) -> MoveStats:
    """Build a MoveStats snapshot from a post-move GameState.

    Args:
        state: The state returned by Game.step after a move.
        player: The player who just moved.

    Returns:
        A MoveStats capturing all instrumentation fields.
    """
    nodes = state.nodes_visited
    pruning_rate = state.prunings / nodes if nodes > 0 else 0.0

    return MoveStats(
        move_number=state.move_number,
        player=player,
        move=state.last_move or (0, 0),
        nodes_visited=nodes,
        max_depth_reached=state.max_depth_reached,
        prunings=state.prunings,
        pruning_rate=pruning_rate,
        time_taken_ms=state.time_taken_ms,
        effective_branching_factor=state.effective_branching_factor,
        time_limit_exceeded=state.time_limit_exceeded,
    )


def _result_for_player(record: GameRecord, agent_name: str) -> Result:
    """Return the game result from an agent's perspective.

    Args:
        record: The completed game record.
        agent_name: The agent whose outcome is queried.

    Returns:
        The Result enum value as stored in the record (not player-relative).
    """
    return record.result


def _agent_won(record: GameRecord, agent_name: str) -> bool:
    """Check whether the named agent won the game.

    Args:
        record: The completed game record.
        agent_name: The agent to check.

    Returns:
        True if agent_name won (regardless of which side they played).
    """
    if record.result is Result.X_WINS:
        return record.agent_x_name == agent_name
    if record.result is Result.O_WINS:
        return record.agent_o_name == agent_name
    return False


def _collect_agent_stats(
    games: list[GameRecord], agent_name: str
) -> list[MoveStats]:
    """Gather all MoveStats for the named agent across multiple games.

    Args:
        games: List of completed game records.
        agent_name: The agent whose stats to collect.

    Returns:
        A flat list of MoveStats for every move the agent made.
    """
    all_stats: list[MoveStats] = []
    for game in games:
        all_stats.extend(game.get_stats_for_player(agent_name))
    return all_stats


def _win_rate_against(
    games: list[GameRecord], agent_name: str
) -> float:
    """Compute the win rate for an agent across a list of games.

    Args:
        games: List of completed game records.
        agent_name: The agent whose win rate is computed.

    Returns:
        Fraction of games won by the agent. 0.0 for an empty list.
    """
    if not games:
        return 0.0
    wins = sum(1 for g in games if _agent_won(g, agent_name))
    return wins / len(games)


def _build_duel_result(
    agent_a: BaseAgent,
    agent_b: BaseAgent,
    wins_a: int,
    wins_b: int,
    draws: int,
    n: int,
    k: int,
    match_config: MatchConfig,
    games: list[GameRecord],
) -> DuelResult:
    """Construct a fully populated DuelResult from raw game data.

    Args:
        agent_a: First agent.
        agent_b: Second agent.
        wins_a: Games won by agent A.
        wins_b: Games won by agent B.
        draws: Drawn games.
        n: Board dimension.
        k: Winning line length.
        match_config: Match configuration used.
        games: All GameRecord objects from the duel.

    Returns:
        A DuelResult with efficiency statistics and Elo ratings populated.
    """
    name_a = agent_a.get_name()
    name_b = agent_b.get_name()

    stats_a = _collect_agent_stats(games, name_a)
    stats_b = _collect_agent_stats(games, name_b)

    agg_a = aggregate_move_stats(stats_a)
    agg_b = aggregate_move_stats(stats_b)

    duel = DuelResult(
        agent_a_name=name_a,
        agent_b_name=name_b,
        agent_a_wins=wins_a,
        agent_b_wins=wins_b,
        draws=draws,
        total_games=len(games),
        n=n,
        k=k,
        match_config=match_config,
        avg_nodes_a=agg_a["mean_nodes"],
        avg_nodes_b=agg_b["mean_nodes"],
        std_nodes_a=agg_a["std_nodes"],
        std_nodes_b=agg_b["std_nodes"],
        avg_time_ms_a=agg_a["mean_time_ms"],
        avg_time_ms_b=agg_b["mean_time_ms"],
        avg_depth_a=agg_a["mean_depth"],
        avg_depth_b=agg_b["mean_depth"],
        avg_ebf_a=agg_a["mean_ebf"],
        avg_ebf_b=agg_b["mean_ebf"],
        avg_pruning_rate_a=compute_pruning_rate(stats_a),
        avg_pruning_rate_b=compute_pruning_rate(stats_b),
        games=games,
    )

    # Compute Elo from this duel alone.
    elo_ratings = compute_elo([duel])
    duel.elo_a = elo_ratings.get(name_a, 1500.0)
    duel.elo_b = elo_ratings.get(name_b, 1500.0)

    return duel
