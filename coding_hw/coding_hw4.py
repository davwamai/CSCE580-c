from typing import List, Dict, Optional, Tuple, Callable
from environments.environment_abstract import Environment, State
from environments.farm_grid_world import FarmGridWorld
from visualizer.farm_visualizer import InteractiveFarm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch import Tensor


class mnistnet(nn.Module):
    def __init__(self):
        super(mnistnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_nnet(train_input_np: np.ndarray, train_labels_np: np.array, val_input_np: np.ndarray,
               val_labels_np: np.array) -> nn.Module:

    train_input = torch.tensor(train_input_np.reshape(-1, 1, 28, 28), dtype=torch.float32)
    train_labels = torch.tensor(train_labels_np, dtype=torch.long)

    val_input = torch.tensor(val_input_np.reshape(-1, 1, 28, 28), dtype=torch.float32)
    val_labels = torch.tensor(val_labels_np, dtype=torch.long)

    train_dataset = TensorDataset(train_input, train_labels)
    val_dataset = TensorDataset(val_input, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

    model = mnistnet()
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()

    epochs = 3
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
        
        evaluate_nnet(model, val_loader)

    return model

def evaluate_nnet(nnet: nn.Module, loader):
    nnet.eval()
    criterion = nn.CrossEntropyLoss()

    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = nnet(data)
            val_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(loader.dataset)
    print(f'\nValidation run: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({100. * correct / len(loader.dataset):.0f}%)\n')



def train_nnet_np(train_input_np: np.ndarray, train_labels_np: np.array, val_input_np: np.ndarray,
                  val_labels_np: np.array) -> Callable:
    """

    :param train_input_np: training inputs
    :param train_labels_np: training labels
    :param val_input_np: validation inputs
    :param val_labels_np: validation labels
    :return: the trained neural network
    """
    pass

def evaluate_nnet_np(nnet: Callable, data_input_np: np.ndarray, data_labels_np: np.array) -> Tuple[float, float]:
    """
    :param nnet: the trained neural network
    :param data_input_np: validation inputs
    :param data_labels_np: validation labels
    :return: the loss and the accuracy
    """
    pass



def update_dp(viz: InteractiveFarm, state_values, policy):
    viz.set_state_values(state_values)
    viz.set_policy(policy)
    viz.window.update()

def update_model_free(viz: InteractiveFarm, state, action_values):
    viz.set_action_values(action_values)
    viz.board.delete(viz.agent_img)
    viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state.agent_idx])[0]
    viz.window.update()

def evaluate_policy(states, state_values, policy, env, discount, policy_eval_cutoff):
    while True:
        delta = 0
        print(f"Iterating through {len(states)} states")
        for state in states:
            v = state_values[state]
            new_v = 0
            actions = env.get_actions(state)
            for action in actions:
                expected_reward, next_states, probs = env.state_action_dynamics(state, action)
                # calculate new value for the state
                new_v += policy[state][action] * sum(probs[i] * (expected_reward + discount * state_values[next_state]) for i, next_state in enumerate(next_states))
            state_values[state] = new_v  # update state value
            delta = max(delta, abs(v - new_v))  # update delta
            print(f"Delta: {delta}")
        if delta <= policy_eval_cutoff:  # check convergence
            break

def improve_policy(states, state_values, policy, env, discount):
    policy_stable = True
    for state in states:
        old_action = policy[state].index(max(policy[state]))  # get action that was previously best
        best_value = -float('inf')
        best_action = None
        actions = env.get_actions(state)
        for action in actions:
            expected_reward, next_states, probs = env.state_action_dynamics(state, action)
            # calculate the value of this action
            action_value = sum(probs[i] * (expected_reward + discount * state_values[next_state]) for i, next_state in enumerate(next_states))
            if action_value > best_value:
                best_value = action_value
                best_action = action
        # update policy to always choose the best action
        for action in actions:
            policy[state][action] = 1 if action == best_action else 0
        if best_action != old_action:
            policy_stable = False  # policy changed
    return policy_stable

 
def policy_iteration(env: FarmGridWorld, states: List[State], state_values: Dict[State, float],
                     policy: Dict[State, List[float]], discount: float, policy_eval_cutoff: float,
                     viz: Optional[InteractiveFarm]) -> Tuple[Dict[State, float], Dict[State, List[float]]]:
    """
    @param env: environment
    @param states: all states in the state space
    @param state_values: dictionary that maps states to values
    @param policy: dictionary that maps states to a list of probabilities of taking each action
    @param discount: the discount factor
    @param policy_eval_cutoff: the cutoff for policy evaluation
    @param viz: optional visualizer

    @return: the state value function and policy found by policy iteration
    """
    print(f"Policy eval cutoff: {policy_eval_cutoff}")
    print(f"Discount: {discount}")
    print(f"States: {len(states)}")
    while True:
        evaluate_policy(states, state_values, policy, env, discount, policy_eval_cutoff)  # evaluate current policy
        if improve_policy(states, state_values, policy, env, discount):  # improve policy based on evaluated values
            break  # policy didn't change ? weve found an optimal policy

    return state_values, policy


def epsilon_greedy_action_selection(state, action_values, epsilon, env):
    """
    Selects an action using an epsilon-greedy policy.
    """
    import random
    if random.random() < epsilon:
        return random.choice(env.get_actions(state))
    else:
        return max(env.get_actions(state), key=lambda a: action_values[state][a])


def sarsa(env: Environment, action_values: Dict[State, List[float]], epsilon: float, learning_rate: float,
          discount: float, num_episodes: int, viz: Optional[InteractiveFarm]) -> Dict[State, List[float]]:
    """
    @param env: environment
    @param action_values: dictionary that maps states to their action values (list of floats)
    @param epsilon: epsilon-greedy policy
    @param learning_rate: learning rate
    @param discount: the discount factor
    @param num_episodes: number of episodes for learning
    @param viz: optional visualizer

    @return: the learned action value function
    """
    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"Episode: {episode}")

        state = env.sample_start_states(3)[0]
        done = False

        action = epsilon_greedy_action_selection(state, action_values, epsilon, env)
        steps = 0

        while not done and steps < 50:
            next_state, reward = env.sample_transition(state, action)

            next_action = epsilon_greedy_action_selection(next_state, action_values, epsilon, env)

            sa_value = action_values[state][action]
            next_sa_value = action_values[next_state][next_action]
            action_values[state][action] = sa_value + learning_rate * (reward + discount * next_sa_value - sa_value)

            # if viz:
            #     update_model_free(viz, state, action_values)

            if env.is_terminal(next_state):
                done = True
                continue

            # S <- S', A <- A'
            state, action = next_state, next_action

    return action_values
