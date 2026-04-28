from src.agents.bns_agent import BNSAgent
from src.core.state import State

agents = {p: BNSAgent(max_depth=9) for p in ("X", "O")}
state = State(n=4, k=3)

while not state.is_terminal():
    player = state.current_player
    row, col = agents[player.name].act(state)
    state.apply(row, col)
    print(f"Player {player.name} plays ({row}, {col})")
    print(state.board.render())

winner = state.winner()
print("Winner:", winner.name if winner else "Draw")
