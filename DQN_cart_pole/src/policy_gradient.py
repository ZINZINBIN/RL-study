'''Policy Gradient Algorithm
- Structure
(1) Monte-Carlo Method
(2) Monte-Carlo Method with dueling(Q -> Q - V(s,a))
(3) Actor-Critic method
(4) Advantage Actor-Critic

- Detail
(1) tensorflow version -> torch version
'''
from itertools import count
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from .model import DQN
from .utility import *
from .buffer import *

class PolicyNet(nn.Module):
    def __init__(self, h, w, output_dims):
        super(PolicyNet, self).__init__()
        self.h = h
        self.w = w
        self.output_dims = output_dims
        self.head = DQN(h,w,output_dims)
        self.softmax = torch.nn.Softmax(dim = 0)

    def forward(self, inputs):
        x = self.head(inputs)
        x = self.softmax(x)
        return x

    def get_action_and_logp(self, x):
        a_prob = self.forward(x)

        # policy(a|s) -> sample()을 통해 확률적으로 action 결정, 이후 log_probability와 함께 출력(동적 그래프 계산을 위해 인스턴스 생성)
        m = torch.distributions.Categorical(probs = a_prob)
        action = m.sample()
        logp = m.log_prob(action)

        return action.item(), logp

    def act(self, x):
        action, logp = self.get_action_and_logp(x)
        return action

# Q-network(s,a) -> Value(s)로 차원이 축소, action에 대해 max Q(s,a) = V(s) 
# V(s) 를 추정하기 위한 Neural Network

class ValueNet(nn.Module):
    def __init__(self, h, w, output_dims = 1):
        super(PolicyNet, self).__init__()
        self.h = h
        self.w = w
        self.output_dims = output_dims
        self.head = DQN(h,w,output_dims)

    def forward(self, inputs):
        x = self.head(inputs)
        return x


def update_replay_memory(env, memory, policy, max_num_steps, device):
    env.reset()
    last_screen = get_screen(env)
    crt_screen = get_screen(env)

    state = crt_screen - last_screen
    steps = 0

    memory.init_memory()

    for t in count():
        state = state.to(device)
        action, logp = policy.get_action_and_logp(state)
        next_state, reward, done = env.step(action)
        reward = torch.tensor([reward], device = device)

        last_screen = crt_screen
        crt_screen = get_screen(env)

        if not done:
            next_state = crt_screen - last_screen
        else:
            next_state = None
        
        memory.push(state, action, next_state, reward, logp)

        steps += 1

        if steps >=max_num_steps or done:
            break
    
    return memory

def get_rewards(memory):
    rewards = memory.reward_list



def policy_gradient_process(
    env, episodes = 100, num_traj=10, max_num_steps=1000, gamma=0.98,
    policy_learning_rate=0.01, value_learning_rate=0.01,
    policy_saved_path='mcpg_policy.pt', value_saved_path='mcpg_value.pt',
    device = 'cpu'
    ):

    init_screen = get_screen(env)
    _,_,screen_height, screen_width = init_screen.shape

    n_actions = env.action_space.n

    policy_net = PolicyNet(screen_height, screen_width, n_actions)
    value_net = ValueNet(screen_height, screen_width, 1)
    memory = PGReplayMemory(max_num_steps)

    policy_optimizer = torch.optim.RMSprop(policy_net.parameters(), lr = policy_learning_rate)
    value_optimizer = torch.optim.RMSprop(value_net.parameters(), lr = value_learning_rate)

    policy_net.to(device)
    value_optimizer.to(device)

    mean_return_list = []

    for episode in tqdm(range(episodes)):

        # value_net : gradient(value_net)으로 경사 학습
        # policy_net : J <= [R - V(s)] * gradient(log(policy_net)) 으로 경사 학습