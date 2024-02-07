from environments.connect_four import ConnectFourState, ConnectFour
import numpy as np
import time

check_flag = False
p1 = None

def is_first(state) -> bool:
    return np.all(state.grid == 0)

def make_move(state: ConnectFourState, env: ConnectFour) -> int:
    """

    :param state: the current state
    :param env: the environment
    :return: the action to take
    """
    global check_flag
    global p1

    if not check_flag:
        p1 = is_first(state)
        check_flag = True

    time_limit = 5 
    start_time = time.time()
    end_time = start_time + time_limit
    depth = 1
    best_score = float('-inf')
    best_action = None

    for action in env.get_actions(state):
        temp_state = env.next_state(state, action)
        if env.is_terminal(temp_state):  # check for winning move
            utility = env.utility(temp_state)
            if utility > 0:  
                return action
            elif utility < 0: # L in 1 
                return action

    while time.time() < end_time:
        for action in env.get_actions(state):
            score = minimax_with_alpha_beta_pruning(env.next_state(state, action), env, depth, float('-inf'), float('inf'), False, end_time, p1)
            if time.time() > end_time:
                return best_action if best_action is not None else env.get_actions(state)[0]
            if score > best_score:
                best_score = score
                best_action = action
        depth += 1  

    return best_action if best_action is not None else env.get_actions(state)[0]


def minimax_with_alpha_beta_pruning(state, env, depth, alpha, beta, maximizingPlayer, end_time, p1):
    if time.time() > end_time:
        return heuristic(state, env, p1)

    if env.is_terminal(state) or depth == 0:
        if env.is_terminal(state):
            return env.utility(state)
        else:
            return heuristic(state, env, p1)
    
    if maximizingPlayer:
        maxEval = float('-inf')
        for action in env.get_actions(state):
            eval = minimax_with_alpha_beta_pruning(env.next_state(state, action), env, depth-1, alpha, beta, False, end_time, p1)
            if time.time() > end_time:
                return maxEval  # best found so far
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = float('inf')
        for action in env.get_actions(state):
            eval = minimax_with_alpha_beta_pruning(env.next_state(state, action), env, depth-1, alpha, beta, True, end_time, p1)
            if time.time() > end_time:
                return minEval
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval


def heuristic(state, env, p1):
    score = 1
    grid = state.grid

    weight = max_analysis(state)

    score *= weight

    score += claim(grid, p1)

    #print("final heuristic score: ", score)
    return score


def max_analysis(state):
    weight = 1
    lines = state.get_lines()
    for line in lines:
        if len(line) < 4:
            continue
        continuous = 0
        eq_player_line = line == 1
        for eq_player in eq_player_line:
            if eq_player:
                continuous += 1
            else:
                continuous = 0

            weight *= (1.3 if continuous == 2 else 1)
            weight *= (1.5 if continuous == 3 else 1)

    return weight


def claim(grid, p1):
    score = 0
    rows, cols = grid.shape
    player = (1 if p1 else -1)

    for col in range(cols):
        for row in range(rows):
            if grid[row, col] == player:
                if row % 2 == 0:
                    score += 10  # favor even rows more
                else:
                    score += 5
            elif grid[row, col] == -player:
                if row % 2 == 0:
                    score -= 10
                else:
                    score -= 5

    center_col = cols // 2
    for row in range(rows):
        if grid[row, center_col] == player:
            score += 20  

    return score

