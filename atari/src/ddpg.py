import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from pytorch_model_summary import summary
import numpy as np
import gym
import random
from gym import Env
from tqdm.auto import tqdm
from collections import namedtuple, deque
from src.utility import get_screen
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
    
    def  sample(self, batch_size : int):
        return random.sample(self.memory, batch_size)

    def get(self):
        return self.memory.pop()

# ddpg network
# use Actor-Critic Algorithm
class Encoder(nn.Module):
    def __init__(self, h : int, w : int):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(64)

        convw = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(w, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        convh = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(h, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        linear_input_dim = convw * convh * 64
        self.linear_input_dim = linear_input_dim

    def _conv2d_size_out(self, size : int, kernel_size : int = 5, stride : int = 2)->int:
        outputs = (size - (kernel_size - 1) - 1) // stride + 1
        return outputs

    def forward(self, x : torch.Tensor)->torch.Tensor:
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

class ActorNetwork(nn.Module):
    def __init__(self, h : int, w : int, n_actions : int, hidden_dims : int = 128):
        super(ActorNetwork, self).__init__()
        self.encoder = Encoder(h,w)
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions

        linear_input_dim =  self.encoder.linear_input_dim

        self.actor = nn.Sequential(
            nn.Linear(linear_input_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, n_actions)
        )

        # init encoder
        self.init_weights()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.encoder(x)

        # Value approximation
        value = self.critic(x)

        # Policy Distribution
        policy = nn.functional.softmax(self.actor(x))
        dist = Categorical(policy)
        return value, dist

    def init_weights(self):
        # xavier init
        torch.nn.init.xavier_uniform_(self.encoder.conv1.weight)
        torch.nn.init.xavier_uniform_(self.encoder.conv2.weight)
        torch.nn.init.xavier_uniform_(self.encoder.conv3.weight)

        nn.init.constant_(self.encoder.conv1.bias, 0.0)
        nn.init.constant_(self.encoder.conv2.bias, 0.0)
        nn.init.constant_(self.encoder.conv3.bias, 0.0)

# Ornstein-Uhlenbeck Process
# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise:
    def __init__(self, action_space, mu : float = 0, theta : float = 0.15, max_sigma : float = 0.3, min_sigma : float = 0.3, decay_period : int = 100000):
        self.mu = mu
        self.theta = theta
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.state = np.empty_like(np.ones(self.action_dim))
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def evolve_state(self):
        x = self.state
        dx  = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t = 0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

# optimize algorithm
