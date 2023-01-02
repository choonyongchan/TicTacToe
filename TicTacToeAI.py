from enum import Enum
import math
from random import shuffle

class Player(Enum):
    O = 'O'
    X = 'X'

class Board:

    def __init__(self, size:int = 3):   
        
        def create_board(size:int) -> list[None]:
            '''Create a new board'''
            return [None]*(size**2)        

        self.size:int = size
        self.board:list[Player] = create_board(size)

    def place(self, idx:int, player:Player):
        '''Place player on the board'''

        def duplicate_board() -> Board:
            board_new:Board = Board(self.size)
            board_new.board:list[int] = self.board.copy()
            return board_new

        board_new:Board = duplicate_board()
        board_new.board[idx] = player  
        return board_new 

    def check_win(self, player:Player) -> bool:
        '''Check if player wins the game'''

        def check_line(line:list[Player]) -> bool:
            '''Check if all values in line belongs to Player'''
            return all(player == elem for elem in line)

        verticals:list[list[int]] = [self.board[i::self.size] for i in range(0,self.size)]
        horizontals:list[list[int]] = [self.board[i:i+self.size] for i in range(0,self.size**2,self.size)]
        diagonals:list[list[int]] = [self.board[::self.size+1],  self.board[self.size-1:self.size**2-1:self.size-1]]

        all_lines:list[list[int]] = verticals + horizontals + diagonals
        is_win:bool = any(check_line(l) for l in all_lines)
        return is_win

    def get_num_empty(self) -> int:
        '''Find number of empty spaces in board'''
        return self.board.count(None)

    def get_grid_empty(self) -> list[int]:
        '''Find which grid in the board is empty'''
        return [i for i,v in enumerate(self.board) if v == None]
    
    def is_no_more_move(self) -> bool:
        return (self.get_num_empty() == 0)

class AI:

    def is_terminal(board:Board) -> int:
        '''Check if the state is a terminal state'''
        is_no_more_move:bool = board.is_no_more_move()
        is_win:bool = any(board.check_win(p) for p in Player)
        return is_no_more_move or is_win

    def evaluate(board:Board) -> int:
        '''Return the value of terminal state'''
        if board.is_no_more_move():
            return 0 # Tie
        reward:int = board.get_num_empty() + 1
        return reward if board.check_win(Player.O) else -reward

    def get_opponent(player:Player):
        return Player.X if player == Player.O else Player.O

    def minimax(board:Board, player:Player, alpha:float = -math.inf, beta:float = math.inf) -> tuple[int, int]:

        def play_maximise():
            nonlocal alpha, beta
            reward_best:int = -math.inf
            grid_best:int = None

            for grid in board.get_grid_empty():
                board_next:Board = board.place(grid, player)
                reward_next, _ = AI.minimax(board_next, AI.get_opponent(player), alpha, beta)
                if reward_next > reward_best:
                    reward_best = reward_next
                    grid_best = grid
                    alpha = max(alpha, reward_best)
                    if beta <= alpha:
                        break

            return reward_best, grid_best
        
        def play_minimise():
            nonlocal alpha, beta
            reward_best:int = math.inf
            grid_best:int = None

            for grid in board.get_grid_empty():
                board_next:Board = board.place(grid, player)
                reward_next, _ = AI.minimax(board_next, AI.get_opponent(player), alpha, beta)
                if reward_next < reward_best:
                    reward_best = reward_next
                    grid_best = grid
                    beta = min(beta, reward_best)
                    if beta <= alpha:
                        break

            return reward_best, grid_best

        if AI.is_terminal(board):
            return AI.evaluate(board), None

        return play_maximise() if player == Player.O else play_minimise()

class Game:

    def __init__(self):
        self.player:Player = Player.O
        self.board:Board = Board()

    def switch(self):
        self.player = (Player.X if self.player == Player.O else Player.O)

    def play(self):

        def place(player:Player, idx:int = None):
            if idx == None:
                raw:str = input("Where do you wish to place your character? ").strip()
                idx:int = int(raw)

            self.board = self.board.place(idx, player)

        while True:
            # Your Turn
            if AI.is_terminal(self.board):
                break
            place(Player.O)
            print(self.board.board)
            # AI Turn
            if AI.is_terminal(self.board):
                break
            reward_best, grid_next = AI.minimax(self.board, Player.X)
            place(Player.X, grid_next)
            print(self.board.board)

Game().play()


        

        




    

