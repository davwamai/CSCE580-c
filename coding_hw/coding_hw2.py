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


def negamax(state, depth, alpha, beta, color, env):
    if depth == 0:
        return color * heuristic(state, env)
    elif env.is_terminal(state):
        return env.utility(state)
    
    max_value = -float('inf')
    for action in env.get_actions(state):
        next_state = env.next_state(state, action)
        value = -negamax(next_state, depth - 1, -beta, -alpha, -color, env)
        max_value = max(max_value, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return max_value


def iterative_deepening(state, env, max_time=4):
    best_action = None
    start_time = time.time()

    for depth in range(1, 100):  # Arbitrary large number for depth
        best_value = -float('inf')
        for action in env.get_actions(state):
            next_state = env.next_state(state, action)
            value = -negamax(next_state, depth - 1, -float('inf'), float('inf'), 1, env)
            if value > best_value:
                best_value = value
                best_action = action
            if time.time() - start_time > max_time:
                print("depth reached: ", depth)
                return best_action

    return best_action


def heuristic(state, env):
    score = 0

    for line in state.get_lines():
        score += evaluate_line(line)
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


