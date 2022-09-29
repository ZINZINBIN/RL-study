import torch
import torch.nn as nn
import numpy as np
import gym
import gc
import random
from tqdm.auto import tqdm
from collections import namedtuple, deque
from typing import Optional
from src.utility import get_screen
from itertools import count

# transition
Transition = namedtuple(
    'Transition',
    ('state', 'action','next_state', 'reward', 'done')
)

# save trajectory from buffer
class ReplayBuffer(object):
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

# Normalized Action Space
class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        return action
    
    def _reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        
        return action

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

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.encoder(x)
        x = nn.functional.tanh(self.actor(x))
        return x

class CriticNetwork(nn.Module):
    def __init__(self, h : int, w : int, n_actions : int, hidden_dims : int = 128):
        super(CriticNetwork, self).__init__()
        self.encoder = Encoder(h,w)
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions

        linear_input_dim =  self.encoder.linear_input_dim

        self.critic = nn.Sequential(
            nn.Linear(linear_input_dim + n_actions, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1)
        )

    def forward(self, x:torch.Tensor, action : torch.Tensor)->torch.Tensor:
        x = self.encoder(x)
        x = torch.cat([x, action], dim = 1)
        x = self.critic(x)
        return x

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

# update policy
def update_policy(
    memory : ReplayBuffer, 
    policy_network : nn.Module, 
    value_network : nn.Module, 
    target_policy_network : nn.Module,
    target_value_network : nn.Module,
    value_optimizer : torch.optim.Optimizer,
    policy_optimizer : torch.optim.Optimizer,
    criterion :Optional[nn.Module] = None,
    batch_size : int = 128, 
    gamma : float = 0.99, 
    device : Optional[str] = "cpu",
    min_value : float = -np.inf,
    max_value : float = np.inf,
    tau : float = 1e-2
    ):

    if len(memory) < batch_size:
        return None, None

    if device is None:
        device = "cpu"
    
    if criterion is None:
        criterion = nn.SmoothL1Loss() # Huber Loss
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # 최종 상태가 아닌 경우의 mask
    non_final_mask = torch.tensor(
        tuple(
            map(lambda s : s is not None, batch.next_state)
        ),
        device = device,
        dtype = torch.bool
    )

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    
    q_values = value_network(state_batch, action_batch)
    policy_loss = -q_values.mean()

    next_q_values = torch.zeros(batch_size, device = device)
    next_q_values[non_final_mask] = target_value_network(non_final_next_states, target_policy_network(non_final_next_states).detach())
    
    bellman_q_values = reward_batch + gamma * next_q_values
    bellman_q_values = bellman_q_values.unsqueeze(1)
    bellman_q_values = torch.clamp(bellman_q_values, min_value, max_value).detach()

    value_loss = criterion(q_values, bellman_q_values)

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    # gradient clipping for value_network and policy_network
    for param in policy_network.parameters():
        param.grad.data.clamp_(-1,1) 
    
    for param in value_network.parameters():
        param.grad.data.clamp_(-1,1) 

    # target network soft tau update
    for target_param, param in zip(target_value_network.parameters(), value_network.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

    for target_param, param in zip(target_policy_network.parameters(), policy_network.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def train_ddpg(
    env,
    memory : ReplayBuffer, 
    policy_network : nn.Module, 
    value_network : nn.Module, 
    target_policy_network : nn.Module,
    target_value_network : nn.Module,
    policy_optimizer : torch.optim.Optimizer,
    value_optimizer : torch.optim.Optimizer,
    value_loss :Optional[nn.Module] = None,
    batch_size : int = 128, 
    gamma : float = 0.99, 
    device : Optional[str] = "cpu",
    min_value : float = -np.inf,
    max_value : float = np.inf,
    tau : float = 1e-2,
    num_episode : int = 256,  
    ):

    if device is None:
        device = "cpu"

    episode_durations = []
    reward_list = []

    for i_episode in tqdm(range(num_episode)):
        env.reset()
        state = get_screen(env)

        mean_reward = []

        for t in count():
            state = state.to(device)
            action = policy_network(state)
            _, reward, done, _ = env.step(action.item())

            mean_reward.append(reward)

            reward = torch.tensor([reward], device = device)

            if not done:
                next_state = get_screen(env)

            else:
                next_state = None
            
            # memory에 transition 저장
            memory.push(state, action, next_state, reward, done)

            state = next_state

            update_policy(
                memory,
                policy_network,
                value_network,
                target_policy_network,
                target_value_network,
                value_optimizer,
                policy_optimizer,
                value_loss,
                batch_size,
                gamma,
                device,
                min_value,
                max_value,
                tau
            )

            if done:
                episode_durations.append(t+1)
                mean_reward = np.mean(mean_reward)
                break

        reward_list.append(mean_reward) 

        # memory cache delete
        gc.collect()

        # torch cache delete
        torch.cuda.empty_cache()

    print("training policy network and target network done....!")
    env.close()

    return episode_durations, reward_list