from enum import Enum
import math
from random import shuffle

class Player(Enum):
    O = 'O'
    X = 'X'

class Board:

    def __init__(self, size:int = 3, board = None):
        self.size:int = size
        self.board:list[int] = (board.copy() if board else Board.create_board(size))

    def create_board(size:int) -> list[None]:
        '''Create a new board'''
        return [None]*(size**2)

    def place(self, grid:int, player:Player) -> None:
        '''Place player on the board'''
        self.board[grid] = player

    def duplicate(self):
        '''Return a new copy of the board'''
        return Board(self.size, self.board)

    def check_win(self, player:Player) -> bool:
        '''Check if player wins the game'''

        def check_line(line:list[Player]) -> bool:
            '''Check if all values in line belongs to Player'''
            return all(player == elem for elem in line)

        # Check verticals
        verticals:list[list[int]] = [self.board[i::self.size] for i in range(0,self.size)]
        # Check horizontals
        horizontals:list[list[int]] = [self.board[i:i+self.size] for i in range(0,self.size**2,self.size)]
        # Check diagonals
        diagonals:list[list[int]] = [self.board[::self.size+1],  self.board[self.size-1:self.size**2-1:self.size-1]]

        all_lines:list[list[int]] = verticals + horizontals + diagonals
        return any(check_line(l) for l in all_lines)

    def find_num_empty(self) -> int:
        '''Find number of empty spaces in board'''
        return self.board.count(None)

    def find_empty_grids(self) -> list[int]:
        '''Find which grid in the board is empty'''
        grids:list[int] = [i for i,v in enumerate(self.board) if v == None]
        shuffle(grids)
        return grids

    def display(self) -> str:
        '''Generate a human friendly TicTacToe board'''
        b:list[int] = [(e if e != None else "_") for e in self.board]
        msg:str = f"""
            Current Board:
            {b[0]}|{b[1]}|{b[2]}
            {b[3]}|{b[4]}|{b[5]}
            {b[6]}|{b[7]}|{b[8]}
        """
        return msg

    def end(self) -> bool:
        '''Return if there is no more possible moves'''
        return (self.find_num_empty() == 0)

class AI:

    def minimax(board:Board, player:Player, alpha:float = -math.inf, beta:float = math.inf) -> tuple[int, Board]:

        def evaluate() -> int:
            '''Calculate the terminal state reward, with a late game penalty.'''
            num_empty:int = board.find_num_empty()
            if num_empty == 0:
                # Stalemate Draw
                return 0
            elif board.check_win(Player.O):
                # O wins (+ve reward)
                return num_empty+1
            elif board.check_win(Player.X):
                # X wins (-ve reward)
                return -(num_empty+1)
            return None

        def is_terminal(val:int) -> bool:
            return (val != None)

        def switch(player:Player) -> Player:
            return Player.X if player == Player.O else Player.O

        def play_maximise(player:Player):
            nonlocal alpha, beta
            reward_best:float = -math.inf
            board_best:Board = None
            grids:list[int] = board.find_empty_grids()
            for grid in grids:
                board_next:Board = board.duplicate()
                board_next.place(grid, player)
                reward_next, _ = AI.minimax(board_next, switch(player), alpha, beta)
                if reward_next > reward_best:
                    reward_best = reward_next
                    board_best = board_next
                    alpha = max(alpha, reward_best)
                if beta <= alpha:
                    break
            return reward_best, board_best
        
        def play_minimise(player:Player):
            nonlocal alpha, beta
            reward_best:float = math.inf
            board_best:Board = None
            grids:list[int] = board.find_empty_grids()
            for grid in grids:
                board_next:Board = board.duplicate()
                board_next.place(grid, player)
                reward_next, _ = AI.minimax(board_next, switch(player), alpha, beta)
                if reward_next < reward_best:
                    reward_best = reward_next
                    board_best = board_next
                    beta = min(beta, reward_best)
                if beta <= alpha:
                    break
            return reward_best, board_best

        # Terminal State
        curr_val:int = evaluate()
        if is_terminal(curr_val):
            return curr_val, board

        return play_maximise(player) if player == Player.O else play_minimise(player)

class Game:

    def __init__(self):
        self.setup()

    def setup(self):

        def get_player():
            while True:
                raw = input("Which character would you like to play as? (O or X) ")
                if raw == "O":
                    return Player.O
                elif raw == "X":
                    return Player.X

        def get_start_first():
            while True:
                raw = input("Would you like to start first? (Y or N) ")
                if raw == "Y":
                    return True
                elif raw == "N":
                    return False
                
        def get_size():
            while True:
                raw = int(input("What board size would you like to play (Default: 3) "))
                return raw if raw else 3

        self.player:Player = get_player()
        self.start_first:bool = get_start_first()
        self.board:Board = Board(size = get_size())

    def switch(self):
        self.player = (Player.X if self.player == Player.O else Player.O)

    def play(self):

        def place():
            raw:str = input("Where do you wish to place your character? (Row Column) ").strip()
            raws:list[int] = [int(r) for r in raw.split()]
            row:int = raws[0]
            col:int = raws[1]

            grid_idx:int = row*self.board.size+col
            self.board.place(grid_idx, self.player)

        if self.start_first:
            place()
            print(self.board.display())

        while not self.board.end():
            # AI turn
            self.switch()
            self.board = AI.minimax(self.board, self.player)[1]
            print(self.board.display())
            self.switch()
            # Your turn
            place()
            print(self.board.display())

Game().play()


        

        




    

