import numpy as np

BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (0,3)
LOSE_STATE = (1,3)
BLOCKED_STATE = (1,1)
START = (2,0)
DETERMINISTIC = False
ACTIONS = ['U', 'L', 'R', 'D']

class State(object):
    def __init__(self, state = START):
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self):
        if self.state == WIN_STATE:
            return 1
        elif self.state == LOSE_STATE:
            return -1
        else:
            return 0
        
    def isEndFunc(self):
        if self.state == WIN_STATE or self.state == LOSE_STATE:
            self.isEnd = True

    def _chooseActionProb(self, action):
        if action == 'U':
            return np.random.choice(["U","L","R"], p = [0.4, 0.3, 0.3])
        elif action == 'D':
            return np.random.choice(["D","L","R"], p = [0.4, 0.3, 0.3])
        elif action == 'R':
            return np.random.choice(["U","D","R"], p = [0.3, 0.3, 0.4])
        elif action == 'L':
            return np.random.choice(["U","D","L"], p = [0.3, 0.3, 0.4])
        else:
            ValueError("action should be one of values : 'U', 'R', 'L', 'D'....")

    def nxtPosition(self, action):
        if self.determine:
            if action == 'U':
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == 'D':
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == 'L':
                nxtState = (self.state[0], self.state[1] - 1)
            elif action == 'R':
                nxtState = (self.state[0], self.state[1] + 1)
            else:
                ValueError("action should be one of values : 'U', 'R', 'L', 'D'....")

            self.determine = False

        else:
            action = self._chooseActionProb(action)
            self.determine = True
            nxtState = self.nxtPosition(action)

        if (nxtState[0] >= 0) and (nxtState[0] <= 2):
            if (nxtState[1] >= 0) and (nxtState[1] <= 3):
                if nxtState != BLOCKED_STATE:
                    return nxtState
                
        return self.state


class Agent(object):
    def __init__(self):
        self.states = [] # 위치와 행동을 기록
        self.actions = ACTIONS
        self.State = State()
        self.isEnd = self.State.isEnd
        self.lr = 0.1
        self.decay_gamma = 0.95

        self.Q_values = {}

        # Q(s,a) initialize
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i,j)] = {}
                for a in self.actions:
                    self.Q_values[(i,j)][a] = 0

    def chooseAction(self):
        max_nxt_reward = 0
        action = 'U'

        for a in self.actions:
            current_position = self.State.state
            nxt_reward = self.Q_values[current_position][a]

            if nxt_reward > max_nxt_reward:
                action = a
                max_nxt_reward = nxt_reward

        print("current pos : {}, greedy action : {}".format(self.State.state, action))
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state = position)

    def reset(self):
        self.states = []
        self.State = State()
        self.isEnd = self.State.isEnd

    def play(self, episodes = 10):
        
        n_iter = 0

        while n_iter < episodes:

            if self.State.isEnd: # 보상 업데이트
                reward = self.State.giveReward()
                for a in self.actions:
                    self.Q_values[self.State.state][a] = reward
                print("Game End Reward : ", reward)

                for s in reversed(self.states):
                    current_q_value = self.Q_values[s[0]][s[1]] # s = (state, action)
                    reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)

                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                    
                self.reset()

                n_iter += 1

            else: # 정책 학습 : greedy-algorithm을 적용
                action = self.chooseAction()
                self.states.append([(self.State.state), action])
                print("current position : {} action : {}".format(self.State.state, action))
                self.State = self.takeAction(action)
                self.State.isEndFunc()
                self.isEnd = self.State.isEnd
                print("nxt state : ", self.State.state)
                print("---------------------------------------")

if __name__ == "__main__":
    agent = Agent()

    agent.play(1000)

    print("latest Q-values....\n")
    print(agent.Q_values)    