'''Policy Gradient Algorithm
- Structure
(1) Monte-Carlo Method
(2) Monte-Carlo Method with dueling(Q -> Q - V(s,a))
(3) Actor-Critic method
(4) Advantage Actor-Critic
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
        super(ValueNet, self).__init__()
        self.h = h
        self.w = w
        self.output_dims = output_dims
        self.head = DQN(h,w,output_dims)

    def forward(self, inputs):
        x = self.head(inputs)
        return x


def update_replay_memory(env, policy, max_num_steps, device):
    env.reset()
    last_screen = get_screen(env)
    crt_screen = get_screen(env)

    state = crt_screen - last_screen
    steps = 0

    memory = PGReplayMemory(max_num_steps)
    memory.init_memory()

    for t in count():
        state = state.to(device)
        action, logp = policy.get_action_and_logp(state)
        next_state, reward, done, _ = env.step(action)
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
    
    memory.rearrange_list()
    return memory

def get_rewards(memory, gamma):
    rewards = memory.reward_list
    dis_rewards = [gamma ** i * r for i, r in enumerate(rewards)]
    sum_rewards = [sum(dis_rewards[i:]) for i in range(len(dis_rewards))]
    return sum_rewards

# 사실상 A2C : Actor 2 Critic Method
# Value + Policy Network를 이용
def policy_gradient_process(
    env, episodes = 100, num_traj=10, max_num_steps=1000, gamma=0.98,
    policy_learning_rate=0.01, value_learning_rate=0.01,
    policy_saved_path='mcpg_policy.pt', value_saved_path='mcpg_value.pt',
    device = 'cpu',
    epi_verbose = 10
    ):

    init_screen = get_screen(env)
    _,_,screen_height, screen_width = init_screen.shape

    n_actions = env.action_space.n

    policy_net = PolicyNet(screen_height, screen_width, n_actions)
    value_net = ValueNet(screen_height, screen_width, 1)
    

    policy_optimizer = torch.optim.RMSprop(policy_net.parameters(), lr = policy_learning_rate)
    value_optimizer = torch.optim.RMSprop(value_net.parameters(), lr = value_learning_rate)

    policy_net.to(device)
    value_net.to(device)

    mean_return_list = []

    for episode in tqdm(range(episodes)):
        # value_net : gradient(value_net)으로 경사 학습
        # policy_net : J <= [R - V(s)] * gradient(log(policy_net)) 으로 경사 학습
        traj_list = [update_replay_memory(env, policy_net, max_num_steps, device) for _ in range(num_traj)]

        # returns : t = 0 ~ T 까지의 합으로 구성된 보상 리스트
        returns = [get_rewards(memory, gamma) for memory in traj_list]
        
        policy_loss_terms = [
            (-1.0) * traj.logp_list[j] * (returns[i][j] - value_net(traj.state_list[j].to(device))) for i, traj in enumerate(traj_list) for j in range(len(traj.action_list))
        ]

        policy_loss = 1.0 / num_traj * torch.cat(policy_loss_terms).sum()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        value_loss_terms = [
            1.0 / len(traj.action_list) * (value_net(traj.state_list[j].to(device)) - returns[i][j]) ** 2 for i, traj in enumerate(traj_list) for j in range(len(traj.action_list))
        ]

        value_loss = 1.0 / num_traj * torch.cat(value_loss_terms).sum()
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        mean_return = 1.0 / num_traj * sum([traj_returns[0] for traj_returns in returns]).detach().cpu().item()
        mean_return_list.append(mean_return)

        if episode % epi_verbose == 0:
            print("Episode : {} : Mean Return : {}".format(episode, mean_return))
            torch.save(policy_net.state_dict(), policy_saved_path)
            torch.save(value_net.state_dict(), value_saved_path)

    return policy_net, value_net, mean_return_list


# Asynchronus Actor to Critic Method
# Actor : update policy parameters, in direction suggested by critic
# critic : update action-value function parameter w
# Multi -thread used

class A3C_agent(object):
    def __init__(
        self, 
        h : int = None,
        w : int = None,
        n_states : int = None,  
        n_actions : int = None,
        gamma : float = 0.99,
        actor_lr : float = 1e-3,
        critic_lr : float = 1e-3,
        actor = None,
        critic = None,
        actor_optimizer = None,
        critic_optimizer = None,
        thread = 8,
        ):

        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        if actor is None:
            self.actor = PolicyNet(h,w,n_actions)
        else:
            self.actor = actor

        if critic is None:
            self.critic = ValueNet(h,w,n_states)
        else:
            self.critic = critic

        if actor_optimizer is None:
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr = self.actor_lr)
        else:
            self.actor_optimizer = actor_optimizer
        
        if critic_optimizer is None:
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr = self.critic_lr)
        else:
            self.critic_optimizer = critic_optimizer

        self.thread = thread