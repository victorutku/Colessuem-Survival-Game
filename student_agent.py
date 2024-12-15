# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import math
import random


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.initial_step = True
        self.autoplay = True
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1)) 
        # Same logic as world.py
        self.MCST = None

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        if(self.initial_step):
            step = None
            state = self.MCST.root_state
            possible_steps = state.validSteps()
            self.MCST = MonteCarloTreeSearch(deepcopy(Simulations(chess_board, my_pos, adv_pos, max_step)))

            # verify if this is our agent's initial action. If this is the case, we will build an MCTS object 
            # with a state that represents the current chessboard position, our position, the position of our opponent, 
            # and the maximum step that may be taken

            for x in possible_steps: # Check if the game terminates
                temps = deepcopy(state)
                temps.playCurrent(x)
                endgame, _, _, winner = temps.check_endgame()
                if(endgame and winner == 1):
                    direction = x
                    my_pos = x
                    return x
            self.MCST.searchTime(1.9)
            direction = step = self.MCST.beststep()
            my_pos =self.MCST.beststep()
            state.playCurrent(step)
            self.initial_step = False
        else:
            new_direction = None
            for i in range(4):
                if(self.MCST.root_state.chess_board[adv_pos[0]][adv_pos[1]][i] != chess_board[adv_pos[0]][adv_pos[1]][i]):
                    new_direction = i
            opp_move = (adv_pos, new_direction)
            self.MCST.currentMove(opp_move)
            possible_steps = self.MCST.root_state.validSteps()
            for x in possible_steps:
                temps = deepcopy(self.MCST.root_state)
                temps.playCurrent(x)
                endgame, _, _, winner = temps.check_endgame()
                if(endgame and winner == 1):
                    my_pos, direction = x
                    return x

            self.MCST.searchTime(1.9)
            my_pos = step = self.MCST.beststep()
            direction = step = self.MCST.beststep()
            self.MCST.currentMove(step)
        
        return my_pos, direction
    
    # The main functionality of this class is to execute the Monte Carlo Tree Search algorithm. 
    # Node keeps track of the visited nodes, its parents, and its children
class MCTSNode:
     def __init__(self, move: tuple = None, parent_node: object = None):
        self.visited = 0
        self.reward = 0
        self.children = {}
        self.move = move
        self.parent = parent_node
        self.result = None 
     def add_node(self, children):
        for child in children:
            self.children[child.move] = child
    
     @property
     def calculateValue(self, explore = 0.5):
        if self.visited == 0:
            return 0 if explore == 0 else sys.maxsize
        else:
            return self.reward / self.visited + explore * math.sqrt(2 * math.log(self.parent.N) / self.visited) # Calculating UCT value
        
class Simulations: # Losely based on Monte-Carlo-Tree-Search-Agent-for-the-Game-of-HEX reference is provided in report.pdf
    def __init__(self, chess_board, my_pos, adv_pos, max_step):
     self.size = len(chess_board[0])
     self.chess_board = deepcopy(chess_board)
     self.my_pos = my_pos
     self.adv_pos = adv_pos
     self.max_step = max_step
     self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
     # if it's my turn value must equal to 1 otherwise 0
     self.turn_to_play = 1 
     # Get a list of the valid moves to play
    def validSteps(self):
        val_steps = []
        if self.turn_to_play == 1:
            position = self.my_pos
            position_x = position
            posiiton_y = position
            opposite_position = self.adv_pos
        else:
            position = self.adv_pos
            position_x = position
            posiiton_y = position 
            opposite_position = self.my_pos

        for x in range(0, self.max_step+1):
            for y in range(0, self.max_step+1-x):
                for barrier in range(4):
                    x_neg = position_x-x # Negative x
                    y_neg = posiiton_y-y # Negative y
                    x_pos = position_x+x # Positive x       
                    y_neg = posiiton_y+y # Positive y
                    if(x == 0):
                        if(y == 0):
                            if self.check_valid_step(position, position, barrier, opposite_position):
                                val_steps.append((position, barrier))
                        else:
                            if(y_neg >= 0):
                                final_position = (position_x, y_neg)
                                if self.check_valid_step(position, final_position, barrier, opposite_position):
                                    val_steps.append((final_position, barrier))
                            if(x_pos < self.size):
                                final_position = (position_x, x_pos)
                                if self.check_valid_step(position, final_position, barrier, opposite_position):
                                    val_steps.append((final_position, barrier))
                    else:
                        if(x_neg >= 0):
                            if(y == 0):
                                final_position = (x_neg, posiiton_y)
                                if self.check_valid_step(position, final_position, barrier, opposite_position):
                                    val_steps.append((final_position, barrier))
                            else:
                                if(y_neg >= 0):
                                    final_position = (x_neg, y_neg)
                                    if self.check_valid_step(position, final_position, barrier, opposite_position):
                                        val_steps.append((final_position, barrier))
                                if(x_pos < self.size):
                                    final_position = (x_neg, x_pos)
                                    if self.check_valid_step(position, final_position, barrier, opposite_position):
                                        val_steps.append((final_position, barrier))
                        if(x_pos < self.size):
                            if(y == 0):
                                final_position = (x_neg, posiiton_y)
                                if self.check_valid_step(position, final_position, barrier, opposite_position):
                                    val_steps.append((final_position, barrier))
                            else:
                                if(y_neg >= 0):
                                    final_position = (x_pos, y_neg)
                                    if self.check_valid_step(position, final_position, barrier, opposite_position):
                                        val_steps.append((final_position, barrier))
                                if(x_pos < self.size):
                                    final_position = (x_pos, x_pos)
                                    if self.check_valid_step(position, final_position, barrier, opposite_position):
                                        val_steps.append((final_position, barrier))

        return val_steps
    # Move for the current player
    def play(self, move):
        if self.turn_to_play == 1:
            self.myMove(move)
            self.turn_to_play = 0
        else: 
            self.ennemyMove(move)
            self.turn_to_play = 1
    
    # Current player's move
    def playCurrent(self, move):
        if self.turn_to_play == 1:
            self.myMove(move)
            self.turn_to_play = 0
        else: 
            self.ennemyMove(move)
            self.turn_to_play = 1

    # My Player's move
    def myMove(self, move):
        position = move
        direction = move
        x = position
        y = position
        self.chess_board[x][y][direction] = True
        self.my_pos = position

    def ennemyMove(self, move):
        position = move
        direction = move
        ennX = position
        ennY = position
        self.chess_board[ennX][ennY][direction] = True
        self.adv_pos = position
    # Helper funciton from world.py
    def check_valid_step(self, start_pos, end_pos, barrier_dir, enn_pos):
        # Endpoint already has barrier or is border
        r, c = end_pos
        if self.chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # Get position of the adversary
        # adv_pos = self.p0_pos if self.turn else self.p1_pos

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if self.chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, enn_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, enn_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached
    #Helper function from world.py modified
    def check_endgame(self):
        # Union-Find
        father = dict()
        for r in range(self.size):
            for c in range(self.size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.size):
            for c in range(self.size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.size):
            for c in range(self.size):
                find((r, c))
        p0_r = find(tuple(self.my_pos))
        p1_r = find(tuple(self.adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        player_win = None
        if p0_r == p1_r:
            return False, p0_score, p1_score, player_win
        if p0_score > p1_score:
            player_win = 0
        elif p0_score < p1_score:
            player_win = 1
        else:
            player_win = -1  # Tie
        return True, p0_score, p1_score
    
class MonteCarloTreeSearch: # Losely based on Monte-Carlo-Tree-Search-Agent-for-the-Game-of-HEX reference is provided in report.pdf
    def __init__(self, state):
        self.runtime = 0
        self.num_nodes = 0
        self.num_rollouts = 0
        self.root_state = deepcopy(state)
        self.root = MCTSNode()
    #The main functionality of this function is choosing a move that is slightly better than the random moves.
    def heuristic(self, moves, state):
        preferable_moves = []
        inward_moves = []  # Nonborder moves
        for x in moves:
            temp = deepcopy(state)
            player = temp.turn_to_play
            temp.play(x)
            game_terminated, _, _, winner = temp.check_endgame()
            # I determined whether there was a move that would allow me to end the game. If so, I would return that move
            if(game_terminated == True):
                if(winner == player):
                    return x
                else: 
                    moves.remove(x)
            else:
                num_walls = sum(bool(x) for x in temp.chess_board[x[0][0]][x[0][1]])
                if(num_walls < 3):
                    preferable_moves.append(x)
                if(x[0][0] != 0 and x[0][0] != state.size-1 and x[0][1] != 0 and x[0][1] != state.size-1):
                    inward_moves.append(x)
        if(len(preferable_moves) != 0):
            return random.choice(preferable_moves)    
        elif(len(inward_moves) != 0):
            return random.choice(inward_moves)
        else:
            if(len(moves) == 1):
                return moves[0]
            else:
                return random.choice(moves)
    #check if game is termianted if not adds new children
    def addChildren(self, parent, state):
        children = []
        if state.check_endgame()[0]:
            return False
        
        for move in state.validSteps():
            children.append(MCTSNode(move, parent))
        parent.addChildren(children)
        return True
    
    def currentMove(self, move):
        if move in self.root.children:
            child = self.root.children[move]
            child.parent = None
            self.root = child
            self.root_state.play(child.move)
            return self.root_state.play(move)
        self.root = MCTSNode()
        
    def getNode(self):
        node = self.root
        state = deepcopy(self.root_state)
        while len(node.children) != 0:  # while we aren't on a leaf node
            children = node.children.values()
            maxValue = max(children, key=lambda n: n.value).value
            nodes = []
            for n in children:
                if n.value == maxValue:
                    nodes.append(n)

            if len(nodes) == 1:
                node = nodes[0]
            else:
                moves = []
                for n in nodes:
                    moves.append(n.move)
                bestMove = self.heuristic(moves, state)
                for n in nodes:
                    if n.move == bestMove:
                        node = n
                        break
            state.play(node.move)
            if node.visited == 0:
                return node, state

        if self.addChildren(node, state):   # If leaf is reached, then generate its children and return it
            childrenValues = list(node.children.values())
            if len(childrenValues) != 0:
                if len(childrenValues) == 1:
                    node = childrenValues[0]
                else:
                    moves = []
                    for child in childrenValues:
                        moves.append(child.move)
                    bestMove = self.heuristic(moves, state)
                    for child in childrenValues:
                        if child.move == bestMove:
                            node = child
                            break
                state.play(node.move)
        return node, state
    
    def getListCurr(self, state):        
        # While game is not finished
        while state.check_endgame()[0] == False:
            # get list of moves for current player
            moves = state.validSteps()
            if(len(moves) == 0):
                return 0 if state.turn_to_play == 1 else 1
            if len(moves) == 1:
                move = moves[0]
            else:
                move = self.heuristic(moves, state)
            state.play(move)

        return state.check_endgame()[3]
    # Trying not to exceed 2 second time limit
    def searchTime(self, limit):
        start_time = time()
        numSimulations = 0
        while time() - start_time < limit:
            node, state = self.getNode()
            if(time() - start_time > limit): break
            result = self.getListCurr(state)
            if(time() - start_time > limit): break
            turn = state.turn_to_play
            self.save(node, turn, result)
            numSimulations += 1
        
        self.runtime = time() - start_time
        self.num_rollouts = numSimulations

    
    # Calculate the reward for the player who just played at the node
    def save(self, node, turn, result):
        if result == 0.5: reward = 0.5
        else:
            if result == turn: reward = 0
            else: reward = 1
        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent
            if result == 0.5: reward = 0.5
            else:
                if reward == 1: reward = 0
                else: reward = 1
    # Most simulated node is selected
    def beststep(self):
        if self.root_state.check_endgame()[0] == True:
            return self.root_state.check_endgame()[0] 

        max_value = max(self.root.children.values(), key=lambda n: n.N).N 
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        moves = []
        for node in max_nodes:
            moves.append(node)
        best = self.heuristic(moves, self.root_state)
        for node in max_nodes:
            if node.move == best:
                best_option = node
                break
        self.root = best_option

        return best_option.move