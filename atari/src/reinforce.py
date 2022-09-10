import torch
import torch.nn as nn
import numpy as np
from gym import Env
from tqdm.auto import tqdm
from collections import namedtuple, deque
from src.utility import get_screen

# transition
Transition = namedtuple(
    'Transition',
    ('state', 'action', 'log_prob', 'next_state', 'reward', 'done')
)

# save trajectory from buffer
class Buffer(object):
    def __init__(self, capacity):
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
    state = env.render()
    done = False

    while(done):
        state = state.to(device)
        action = policy.select_action(state).squeeze(0).cpu().numpy()
        log_prob = policy(state).log_prob(action)
        _, reward, done, _ = env.step(action)
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
    loss = torch.Tensor([0], device=device)
    optimizer.zero_grad()

    for log_p, reward in zip(log_p_list, reward_list):
        Gt = Gt * gamma + reward
        loss += log_p * Gt
        loss.backward()

    optimizer.step()


def REINFORCE(
    buffer : Buffer,
    env : Env,
    policy : nn.Module,
    optimizer : torch.optim.Optimizer,
    device : str = 'cpu', 
    gamma : float = 0.95,
    max_frame : int = 1e6,
    verbose_frame : int = 1e2,
    ):



# training process for each episode
    for frame_idx in tqdm(range(max_frame)):
        envs.reset()
        state = envs.render()

        log_probs = []
        rewards = []
        masks = []
        values = []
        losses = []
        entropy = 0

        # multi-steps  A3C algorithm : to avoid strong correlation between samples
        for n_step in range(n_steps):

            state = state.to(device)
            value, dist = a3c(state)
            action = dist.sample()
            _, reward, done, _ = envs.step(action.cpu().numpy())

            reward = torch.from_numpy(reward).to(device)
            state = envs.render()

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward.unsqueeze(1))
            masks.append(torch.FloatTensor(1-done).unsqueeze(1).to(device))

        # test plot
        if frame_idx % 100 == 0:
            test_reward = test_env()
            print("frame_idx : {}, test_reward : {:.3f}".format(frame_idx, test_reward))
            test_rewards.append(test_reward)
            test_frames.append(frame_idx)

        # next_state에 대한 예측값 
        log_probs = torch.cat(log_probs).to(device)
        next_state = state.to(device)
        next_value, next_dist = a3c(next_state)
    
        returns = compute_returns(next_value, rewards, masks, gamma)

        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()

        loss = policy_loss_coeff * policy_loss + value_loss_coeff * value_loss - entropy_loss_coeff * entropy
        optimizer.zero_grad()
        
        # gradient clipping 
        torch.nn.utils.clip_grad_norm_(a3c.parameters(), max_grad_norm)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())