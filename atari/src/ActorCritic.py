import numpy as np
import torch 
from gym import Env
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Union
from src.model import Encoder
from src.utility import get_screen
from pytorch_model_summary import summary
from collections import namedtuple
from itertools import count

# transition
Transition = namedtuple(
    'Transition',
    ('state', 'action', 'log_prob', 'next_state', 'reward', 'done')
)

class Trajectory(object):
    def __init__(self, max_T : int):
        self.max_T = max_T
        self.memory_size = 0
        self.memory = {
            "state":[],
            "action":[],
            "log_prob":[],
            "next_state":[],
            "reward":[],
            "done":[]
        }
        
    def push(self, *args):
        if self.memory_size >= self.max_T:
            return
        else:
            state, action, log_prob, next_state, reward, done = Transition(*args)
            self.memory['state'].append(state)
            self.memory['action'].append(action)
            self.memory['log_prob'].append(log_prob)
            self.memory['next_state'].append(next_state)
            self.memory['reward'].append(reward)
            self.memory['done'].append(done)
            self.memory_size += 1
    
    def clear(self):
        for key, items in self.memory:
            items.clear()
        self.memory_size = 0
    
    def pull(self):
        return Transition(*self.memory.values())
    

# Actor Network : Pi(a|s)
class Actor(nn.Module):
    def __init__(self, h : int, w : int, n_actions : int, hidden_dims : int):
        super(Actor, self).__init__(self)
        self.encoder = Encoder(h,w)
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        
        # pi(s,a)
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.linear_input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, n_actions)
        )
    
    def forward(self, x : torch.Tensor):
        x = self.encoder(x)
        x = self.mlp(x)
        policy = nn.functional.softmax(x)
        dist = Categorical(policy)
        return dist
    
    def action(self, x : Union[torch.Tensor, np.ndarray], device : str = 'cpu'):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).to(device)
        with torch.no_grad():
            x = self.encoder(x)
            x = self.mlp(x)
            policy = nn.functional.softmax(x)
            dist = Categorical(policy).detach()
        return dist
    
# Critic Network : V(s)
class Critic(nn.Module):
    def __init__(self, h : int, w : int, hidden_dims : int):
        super(Critic, self).__init__(self)
        self.encoder = Encoder(h,w)
        self.hidden_dims = hidden_dims
        
        # Q(s,a)
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.linear_input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1)
        )
    
    def forward(self, x : torch.Tensor):
        x = self.encoder(x)
        x = self.mlp(x)
        return x
    
# update trajectory from 1 episode
def update_trajectory(
    env : Env,
    actor : nn.Module,
    critic : nn.Module,
    trajectory : Trajectory,
    max_T : int,
    device : str              
    ):
    
    if trajectory.memory_size != 0:
        trajectory.clear()
        
    state = get_screen(env)
    episode_reward = 0
    episode_duration = 0
    
    for t in count():
        state = state.to(device)
        action = actor.action(state)
        log_prob = actor(state).log_prob(action)
        
        _, reward, done, _ = env.step(action.detach().cpu().numpy())
        
        # update total reward for monitoring
        episode_reward += reward
        episode_duration += 1
        
        reward = torch.tensor([reward], device = device)
        next_state = get_screen(env)
        trajectory.push(state, action, log_prob, next_state, reward, done)
        
        if t >= max_T:
            break
        
        if done:
            break
        else:
            state = next_state
        
    return episode_reward, episode_duration
    
# Policy improvement : optimize actor and critic network
def update_policy(
    actor : nn.Module,
    critic : nn.Module,
    trajectory : Trajectory,
    actor_optim : torch.optim.Optimizer,
    critic_optim : torch.optim.Optimizer,
    critic_loss_fn : nn.Module,
    device : str,
    gamma : float
    ):
    
    states, actions, log_probs, next_states, rewards, dones = trajectory.pull()
    trajectory.clear()
    
    actor.to(device)
    critic.to(device)
    
    actor_loss = 0
    critic_loss = 0
    Gain = []
    discount_reward = torch.Tensor([0])
    
    # compute discounted reward
    for reward in rewards[::-1]:
        discount_reward = reward + gamma * discount_reward
        Gain.append(discount_reward)
    
    Gain = [g for g in reversed(Gain)]
    
    for state, action, log_prob, reward, done in zip(states, actions, log_probs, Gain, dones):
        # compute critic loss
        value = critic(state.to(device))
        critic_loss += critic_loss_fn(reward.to(device), value)
        
        # compute actor loss
        bellman_reward = reward + gamma * value
        actor_loss -= bellman_reward * log_prob
        
    # optimization
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()
    
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()