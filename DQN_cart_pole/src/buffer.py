from collections import namedtuple, deque
import random

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