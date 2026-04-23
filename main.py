from src.agents.minimax_rewards_alphabeta_agent import MinimaxRewardsAlphaBetaAgent
from src.core.state import State
from src.core.types import Player

agents = {p: MinimaxRewardsAlphaBetaAgent(Player[p]) for p in ("X", "O")}
state = State()

while not state.is_terminal():
    player = state.current_player
    row, col = agents[player.name].act(state)
    state.apply(row, col)
    print(f"Player {player.name} plays ({row}, {col})")
    print(state.board.render())

winner = state.winner()
print("Winner:", winner.name if winner else "Draw")
