from collections import deque
import numpy as np
import os
import argparse
import time
import gym
from gym import wrappers, logger
from pyvirtualdisplay import Display
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type = int, default = 100)
parser.add_argument("--demo", type = bool, default = False)
parser.add_argument("--slippery", type = bool, default = False)
parser.add_argument("--decay", type = float, default = 0.9, help = "decay rate for epsilon")
parser.add_argument("--gamma", type = float, default = 0.95, help = "discount rate for reward")
parser.add_argument("--epsilon", type = float, default = 0.8, help = "epsilon for e-greedy algorithm")
parser.add_argument("--epsilon_min", type = float, default = 0.2, help = "epsilon min for e-greedy algorithm")
parser.add_argument("--lr", type = float, default = 0.1, help = "learing rate")
parser.add_argument("--is_explore", type = bool, default = True)
parser.add_argument("--delay", type = float, default = 1e-1)

args = parser.parse_args()

# parameter 
episodes = args.episodes
demo = args.demo
slippery = args.slippery
decay = args.decay
gamma = args.gamma
epsilon = args.epsilon
epsilon_min = args.epsilon_min
learning_rate = args.lr
is_explore = args.is_explore
delay = args.delay

# initalize display
display = Display(visible= False, size = (400,300))
display.start()

env = gym.make("FrozenLake-v1", is_slippery = slippery)
env.reset()

# define spaces
observation_space = env.observation_space
action_space = env.action_space

# agent
class QAgent():
    def __init__(
        self, 
        observation_space, 
        action_space, 
        demo = False, 
        slippery = False, 
        decay = 0.99, 
        gamma = 0.9, 
        epsilon = 0.9,
        epsilon_min = 0.1,
        learning_rate = 0.1):

        self.action_space = action_space
        self.observation_space = observation_space
        self.demo = demo
        self.slippery = slippery
        self.decay = decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate

        col = action_space.n
        row = observation_space.n

        self.q_table = np.zeros((row,col)) # row * col ????????? q table ??????

        self.epsilon_decay = decay
        self.epsilon_min = epsilon_min

        if slippery:
            self.filename = "q-frozenlake-slippery.npy"
        else:
            self.filename = "q-frozenlake.npy"    

        self.demo = demo

        # demo ??????????????? ????????? ???????????? ?????????(????????? ????????? ?????????????????? ??????)
        if demo:
            self.epsilon = 0   

    def act(self, state, is_explore = False):
        # is_explore ????????? e-greedy algorithm??? ????????? ?????? -> action_space ???????????? sample() ????????? ??????
        if is_explore or np.random.rand() < self.epsilon:
            return self.action_space.sample()
        
        # q_table : observation_space.n * action_space.n
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):

        # Q(s,a) <- Q(s,a) + [R + gamma * max(Q(s_next, a_next)) - Q(s,a)] * learning_rate

        q_value = self.gamma * np.amax(self.q_table[next_state])
        q_value += reward
        q_value -= self.q_table[state, action]
        q_value *= self.learning_rate

        q_value += self.q_table[state, action] + q_value

        # update q_table
        self.q_table[state, action] = q_value

    def print_q_table(self):
        print(self.q_table)
        print("Epsilon : {}".format(self.epsilon))
    
    def save_q_table(self):
        np.save(self.filename, self.q_table)
    
    def load_q_table(self):
        self.q_table = np.load(self.filename)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


agent = QAgent(
    observation_space,
    action_space,
    demo,
    slippery,
    decay,
    gamma,
    epsilon,
    epsilon_min,
    learning_rate
)

wins = 0

# training q-table
for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        # Q-table????????? ?????? ??????
        action = agent.act(state, is_explore = is_explore)
        # env?????? action??? ?????? next_state, reward, done, _ ??? ????????????s
        next_state, reward, done, _ = env.step(action)
        # ?????? ?????? ?????????
        os.system('clear')
        # rendering
        env.render()

        if done:
            if reward > 0:
                wins += 1

                if not demo:
                    agent.update_q_table(state, action, reward, next_state)
                    agent.update_epsilon()

                state = next_state
                percent_wins = 100.0 * wins / (episode + 1)
                print("----------- %0.2f%% Goals in %d Episode -----------"%(percent_wins, episode))

                if done:
                    time.sleep(5 * delay)
                else:
                    time.sleep(delay)


print("# Q-network training done....!")
env.close()