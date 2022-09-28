# buffer class
import random
from collections import namedtuple, deque

Transition = namedtuple(
    'Transition',
    ('state', 'compass', 'action', 'next_state', 'next_compass', 'reward', 'done', 'prob_a')
)

class ReplayBuffer(object):
    def __init__(self, T_horizon : int):
        self.capacity = T_horizon
        self.memory = deque([], maxlen = T_horizon)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def get_trajectory(self):
        traj = [self.memory[idx] for idx in range(0, len(self.memory))]
        return traj

    def clear(self):
        self.memory.clear()
        
    def __len__(self):
        return len(self.memory)