from collections import namedtuple, deque
import random
import numpy as np

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
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


# Policy Gradient Method에서 활용할 memory
# (state, action, next_state, reward) -> (state, action, next_state, reward, logp)
class PGReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)
        self.transition = namedtuple(
            'Transition',
            ('state','action','next_state','reward','logp')
        )
    def push(self, *args):
        self.memory.append(self.transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

    def init_memory(self):
        self.memory.clear()

    def rearrange_list(self):

        state_list = []
        action_list = []
        next_state_list = []
        reward_list = []
        logp_list = []

        maxlen = self.__len__()

        for idx in range(maxlen):
            sample = self.memory.popleft()
            state_list.append(sample.state)
            action_list.append(sample.action)
            next_state_list.append(sample.next_state)
            reward_list.append(sample.reward)
            logp_list.append(sample.logp)

        self.state_list = state_list
        self.action_list = action_list
        self.next_state_list = next_state_list
        self.reward_list = reward_list
        self.logp_list = logp_list


# Prioritized Replay Memory
# paper : PRIORITIZED EXPERIENCE REPLAY
# reference : https://arxiv.org/pdf/1511.05952.pdf

class PrioritzedReplayMemory(object):
    def __init__(self, capacity, prob_alpha, transition = Transition):
        self.memory = []
        self.capacity = capacity
        self.prob_alpha = prob_alpha
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype = np.float32)
        self.transition = transition

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        max_priorities = self.priorities.max() if self.__len__() > 0 else 1.0

        if self.__len__() < self.capacity:
            self.memory.append(self.transition(*args))
        else:
            self.memory[self.pos] = self.transition(*args)

        self.priorities[self.pos] = max_priorities
        self.pos = (self.pos + 1) % self.capacity

    def get_weights(self, probs, indices, beta):
        weights = (self.__len__() * probs[indices]) ** (-beta)
        weights /= weights.max()
        return np.array(weights, dtype = np.float32)
    

    def sample(self, batch_size, beta = 0.4):
        if self.__len__() == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        probs = priorities ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(self.__len__(), batch_size, p = probs)
        samples = [self.memory[idx] for idx in indices]
        weights = self.get_weights(probs, indices, beta)

        

