from environments.connect_four import ConnectFourState, ConnectFour
import numpy as np
import time

global move_count
def make_move(state: ConnectFourState, env: ConnectFour) -> int:
    """

    :param state: the current state
    :param env: the environment
    :return: the action to take
    """
    return iterative_deepening(state, env)


# def negamax(state, depth, alpha, beta, color, env):
#     if env.is_terminal(state):
#         return env.utility(state)
#     elif depth == 0:
#         return color * heuristic(state, env)
    
#     max_value = -float('inf')
#     for action in env.get_actions(state):
#         next_state = env.next_state(state, action)
#         value = -negamax(next_state, depth - 1, -beta, -alpha, -color, env)
#         max_value = max(max_value, value)
#         alpha = max(alpha, value)
#         if alpha >= beta:
#             break
#     return max_value

def pvs(state, depth, alpha, beta, color, env):
    if env.is_terminal(state):
        return color * env.utility(state)
    elif depth == 0:
        return color * heuristic(state, env)
    
    max_value = -float('inf')
    first_child = True
    for action in env.get_actions(state):
        next_state = env.next_state(state, action)
        if first_child:
            value = -pvs(next_state, depth - 1, -beta, -alpha, -color, env)
            first_child = False
        else:
            # null window search
            value = -pvs(next_state, depth - 1, -alpha - 1, -alpha, -color, env)
            # if score is within the original alpha-beta window, perform full re-search
            if alpha < value < beta:
                value = -pvs(next_state, depth - 1, -beta, -value, -color, env)
        
        max_value = max(max_value, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return max_value


def iterative_deepening(state, env, max_time=4):
    start_time = time.time()
    depth = 1
    best_move = None
    alpha = -float('inf')
    beta = float('inf')
    
    while True:
        if time.time() - start_time > max_time:
            break
        
        current_best_move, score = deepening_search(state, depth, alpha, beta, 1, env, start_time, max_time)
        
        # update best move if a better one was found
        if current_best_move is not None:
            best_move = current_best_move
        
        # increment depth for the next iteration
        depth += 1
    print("depth reached: ", depth) 
    return best_move

def deepening_search(state, depth, alpha, beta, color, env, start_time, max_time):
    best_move = None
    best_score = -float('inf')
    for action in env.get_actions(state):
        if time.time() - start_time > max_time:
            break
        
        next_state = env.next_state(state, action)
        score = -pvs(next_state, depth - 1, -beta, -alpha, -color, env)
        
        if score > best_score:
            best_score = score
            best_move = action
        
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    
    return best_move, best_score
     
# def iterative_deepening(state, env, max_time=4):
#     best_action = None
#     start_time = time.time()

#     for depth in range(1, 100):  # Arbitrary large number for depth
#         best_value = -float('inf')
#         for action in env.get_actions(state):
#             next_state = env.next_state(state, action)
#             value = -negamax(next_state, depth - 1, -float('inf'), float('inf'), 1, env)
#             if value > best_value:
#                 best_value = value
#                 best_action = action
#             if time.time() - start_time > max_time:
#                 print("depth reached: ", depth)
#                 return best_action

#     return best_action

        
def heuristic(state, env):
    score = 0
    move_count = m_count(state)
    grid = state.grid 

    # Evaluate lines
    for line in state.get_lines():
        score += evaluate_line(line)

    # Determine if bot is first or second player based on move count
    bot_first = move_count % 2 == 0

    # Apply claim_even or claim_odd logic based on who went first
    if bot_first:
        score += claim(state, True, grid)
    else:
        score += claim(state, False, grid)

    return score

def claim(state, is_bot_first, grid):
    score = 0
    
    # Iterate over the grid to favor moves in strategic rows (even or odd)
    for row_idx in range(grid.shape[0]):
        for col_idx in range(grid.shape[1]):
            piece = grid[row_idx][col_idx]
            # Score bonus for controlling strategic rows based on who went first
            if is_bot_first and row_idx % 2 != 0:  # Bot goes first, favor even rows
                score += (10 if piece == 1 else -5)  # Favor bot pieces, penalize opponent pieces in these rows
            elif not is_bot_first and row_idx % 2 == 0:  # Bot goes second, favor odd rows
                score += (10 if piece == 1 else -5)

    return score

def evaluate_line(line):
    score = 0
    continuous_length = 1

    for i in range(1, len(line)):
        if line[i] == line[i - 1] and line[i] != 0:
            continuous_length += 1
        else:
            if continuous_length > 1:  # Only score if there are 2 or more continuous chips
                score += continuous_length**2 * line[i - 1]  # Power of number of continuous chips
            continuous_length = 1

    # Check at the end of the line
    if continuous_length > 1:
        score += continuous_length**2 * line[-1]

    return score

def m_count(state):
    return np.count_nonzero(state.grid)
