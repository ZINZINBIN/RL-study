import torch
import torch.nn as nn
import numpy as np
import gym
from gym import Env
from tqdm.auto import tqdm
from collections import namedtuple, deque
from src.utility import get_screen
from src.model import PolicyNetwork
from pyvirtualdisplay import Display

# transition
Transition = namedtuple(
    'Transition',
    ('state', 'action', 'log_prob', 'next_state', 'reward', 'done')
)

# save trajectory from buffer
class Buffer(object):
    def __init__(self, capacity : int):
        self.memory = deque([], maxlen = capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def __len__(self):
        return len(self.memory)
    def get(self):
        return self.memory.pop()

# accumulate trajectory from environment : 1 episode
def update_trajectory(
    buffer : Buffer, 
    env : Env, 
    policy : nn.Module, 
    device : str = 'cpu'
    ):

    env.reset()
    state = get_screen(env)
    done = False

    while(not done):
        state = state.to(device)
        action = policy.select_action(state).squeeze(0)
        log_prob = policy(state).log_prob(action)
        _, reward, done, _ = env.step(action.detach().cpu().numpy())
        reward = torch.tensor([reward], device = device)
        next_state = get_screen(env)

        buffer.push(state, action, log_prob, next_state, reward, done)

        if done:
            break
        else:
            state = next_state
    
# update policy : optimize policy network with REINFORCE algorithm
def update_policy(
    buffer : Buffer,  
    optimizer : torch.optim.Optimizer,
    device : str = 'cpu', 
    gamma : float = 0.95,
    ):
    
    # unwrapp the trajectory from buffer
    log_p_list = []
    reward_list = []
    max_len = buffer.__len__()

    for _ in range(max_len):
        _, _, log_p, _, reward, _ = buffer.get()
        log_p_list.append(log_p)
        reward_list.append(reward)

    # calculate objective function J using log_p and reward
    Gt = 0
    loss = 0
    optimizer.zero_grad()

    for log_p, reward in zip(log_p_list, reward_list):
        Gt = Gt * gamma + reward
        loss += log_p * Gt
        loss.backward(retain_graph = True)

    optimizer.step()

def evaluate_policy(
    env : Env, 
    policy : nn.Module, 
    device : str = 'cpu'
    ):

    env.reset()
    state = get_screen(env)
    done = False
    total_reward = 0
    n_steps = 0

    while(not done):
        state = state.to(device)
        action = policy.select_action(state).squeeze(0)
        _, reward, done, _ = env.step(action)
        reward = torch.tensor([reward], device = device)
        next_state = get_screen(env)

        n_steps += 1
        total_reward += reward

        if done:
            break
        else:
            state = next_state
    
    total_reward = total_reward.cpu().numpy()

    return total_reward, n_steps

def REINFORCE(
    buffer : Buffer,
    env : Env,
    policy : nn.Module,
    optimizer : torch.optim.Optimizer,
    device : str = 'cpu', 
    gamma : float = 0.95,
    num_episode : int = 1024,
    verbose : int = 8
    ):

    for episode_idx in tqdm(range(num_episode)):

        env.reset()

        # update trajectory for use
        update_trajectory(buffer, env, policy, device)

        # optimize policy network
        update_policy(buffer, optimizer, device, gamma)

        # test for each episode
        if episode_idx % verbose == 0:
            total_reward, n_steps = evaluate_policy(env,  policy, device)
            print("episode : {}, total_reward : {},  n_steps : {}".format(episode_idx+1, total_reward, n_steps))