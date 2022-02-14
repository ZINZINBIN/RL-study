from collections import namedtuple, deque
import random
import numpy as np

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward','done')
)

# Batch 단위로 데이터를 샘플링하여 학습에 필요한 batch 데이터를 생성
# queue 구조를 이용
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)

    def push(self, *args):
        '''
        transition data 저장
        '''
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)