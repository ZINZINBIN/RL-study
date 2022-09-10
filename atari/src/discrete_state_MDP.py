import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import namedtuple, deque
from tqdm import tqdm

class Discrete_MDP_ReplayBuffer(object):
    def __init__(self, capacity : int):
        self.memory = deque([], maxlen = capacity)
        self.transition = namedtuple(
            'transition',(
                "state", "action", "reward", "next_state", "done"
            )
        )
    def push(self, *args):
        self.memory.append(
            self.transition(*args)
        )
    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        sample = self.transition(*zip(*sample))

        state = np.concatenate(sample.state)
        action = sample.action
        reward = sample.reward
        next_state = np.concatenate(sample.next_state)
        done = sample.done

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.memory)

class StochasticMDP(object):
    def __init__(self):
        self.end           = False
        self.current_state = 2
        self.num_actions   = 2
        self.num_states    = 6
        self.p_right       = 0.5

    def reset(self):
        self.end = False
        self.current_state = 2
        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1.
        return state

    def step(self, action):
        if self.current_state != 1:
            if action == 1:
                if random.random() < self.p_right and self.current_state < self.num_states:
                    self.current_state += 1
                else:
                    self.current_state -= 1
                
            if action == 0:
                self.current_state -= 1
            
            if self.current_state == self.end:
                self.end = True
            
        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1.

        if self.current_state == 1:
            if self.end:
                return state, 1.00, True, {}
            else:
                return state, 1.00/100.00, True, {}
        else:
            return state, 0.0, False, {}


class Net(nn.Module):
    def __init__(self, n_inputs : int, n_outputs : int, n_actions : int, hidden : int = 64):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            #nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            #nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_outputs)
        )
        self.n_actions = n_actions
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state  = torch.FloatTensor(state).unsqueeze(0)
            action = self.forward(torch.autograd.Variable(state, volatile=True)).max(1)[1]
            return action.data[0]
        else:
            return random.randrange(self.n_actions)

def num2onehot(x : int):
    one_hot = np.zeros(6)
    one_hot[x-1] = 1.
    return one_hot

def optimize(model : nn.Module, optimizer : torch.optim.Optimizer, buffer : Discrete_MDP_ReplayBuffer, batch_size : int, device : str = 'cpu', gamma : float = 0.99):
    if batch_size > len(buffer):
        return

    state, action, reward, next_state, done = buffer.sample(batch_size)
    
    state = torch.autograd.Variable(torch.FloatTensor(state)).to(device)
    action = torch.autograd.Variable(torch.LongTensor(action)).to(device)
    reward = torch.autograd.Variable(torch.FloatTensor(reward)).to(device)
    next_state = torch.autograd.Variable(torch.FloatTensor(next_state), volatile = True).to(device)
    done = torch.autograd.Variable(torch.FloatTensor(done)).to(device)

    model.to(device)
    model.train()

    optimizer.zero_grad()

    q_value = model(state)
    q_value = q_value.gather(1, action.unsqeeuze(1)).squeeze(1)

    next_q_value = model(next_state).max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1-done)

    loss = (q_value - expected_q_value).pow(2).mean()
    loss.backward()
    optimizer.step()


# epsilon per frame 
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


if __name__ == "__main__":

    # cuda check
    if torch.cuda.is_available():
        print("cuda available : ", torch.cuda.is_available())
        print("cuda device count : ", torch.cuda.device_count())
        device = "cuda:0"
    else:
        device = "cpu" 

    env = StochasticMDP()

    # environment setting
    n_actions = env.num_actions
    n_goals = env.num_states
    n_states = env.num_states

    # learning rate for optimizer
    lr_controller = 1e-3
    lr_meta_controller = 1e-3

    # replay buffer memory
    memory_len = 10000

    # discount
    gamma = 0.99

    # batch
    batch_size = 32

    # model hidden dim
    hidden = 128

    # total episode
    num_episode = 10000

    # controller
    controller = Net(n_inputs = n_states + n_goals, n_outputs = n_actions, n_actions = n_actions, hidden = hidden)
    target = Net(n_inputs = n_states + n_goals, n_outputs = n_actions, n_actions = n_actions, hidden = hidden)

    # meta controller
    meta_controller = Net(n_inputs = n_states, n_outputs = n_actions, n_actions = n_actions, hidden = hidden)
    meta_target = Net(n_inputs = n_states, n_outputs = n_actions, n_actions = n_actions, hidden = hidden)

    # optimizer
    optimizer = torch.optim.SGD(controller.parameters(), lr = lr_controller)
    meta_optimizer = torch.optim.SGD(meta_controller.parameters(), lr = lr_meta_controller)

    # Replay Buffer
    replay_buffer = Discrete_MDP_ReplayBuffer(memory_len)
    meta_replay_buffer = Discrete_MDP_ReplayBuffer(memory_len)

    # training
    state = env.reset()
    done = False
    all_rewards = []
    episode_reward = 0

    for episode in tqdm(range(0, num_episode)):
        print("state : size : ", state.shape)
        goal = meta_controller.act(state, epsilon_by_frame(episode))
        goal_onehot = num2onehot(goal)

        meta_state = state
        extrinsic_reward = 0

        while not done and goal != np.argmax(state):
            goal_state = np.concatenate([state, goal_onehot])
            action = controller.act(goal_state, epsilon_by_frame(episode))

            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            extrinsic_reward += reward

            # s6 방문시 reward : 1, 아닐 경우 0.01
            intrinsic_reward = 1.0 if goal == np.argmax(next_state) else 0

            next_goal_state = np.concatenate([next_state, goal_onehot])
            replay_buffer.push(goal_state, action, intrinsic_reward, next_goal_state, done)

            # update state
            state = next_state

            # optimize controller
            optimize(controller, optimizer, replay_buffer, batch_size, device, gamma)

            # optmize meta controller
            optimize(meta_controller, meta_optimizer, meta_replay_buffer, batch_size, device, gamma)

        # meta controller
        # state : s_0
        # next_state : s_n
        # goal : option

        meta_replay_buffer.push(meta_state, goal, extrinsic_reward, state, done)

        if done:
            state = env.reset()
            done = False
            all_rewards.append(episode_reward)
            episode_reward = 0

    
    import matplotlib.pyplot as plt

    plt.plot(range(1, num_episode + 1), all_rewards, 'b--', label = 'sum reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward(sum)")
    plt.legend()
    plt.savefig("./results/HDQN_for_discrete_MDP_reward.png")