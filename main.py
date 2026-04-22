from src.agents.random_agent import RandomAgent
from src.core.board import Board
from src.core.state import State

agents = {p: RandomAgent() for p in ("X", "O")}
state = State()

while not state.is_terminal():
    player = state.current_player
    row, col = agents[player.name].act(state)
    state.apply(row, col)
    print(f"Player {player.name} plays ({row}, {col})")
    print(Board.render(state.board))

winner = state.winner()
print("Winner:", winner.name if winner else "Draw")
