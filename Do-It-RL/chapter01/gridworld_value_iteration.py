import numpy as np
from typing import Tuple, List

ACTIONS = ('U', 'D', 'L', 'R')
DELTA_THRESHOLD = 1e-3
GAMMA = 0.9
ACCURACY = 0.8

class Grid:
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        self.rewards = rewards
        self.actions = actions
    
    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        return s not in self.actions

    def move(self, action):
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
            
        return self.rewards.get((self.i, self.j), 0)

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())

def initialize_random_policy(grid):
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ACTIONS)
    return policy
        
def standard_grid():
	grid = Grid(3, 4, (2, 0))
	rewards = {(0, 3): 1, (1, 3): -1}
	actions = {
		(0, 0): ('D', 'R'),
		(0, 1): ('L', 'R'),
		(0, 2): ('L', 'D', 'R'),
		(1, 0): ('U', 'D'),
		(1, 2): ('U', 'D', 'R'),
		(2, 0): ('U', 'R'),
		(2, 1): ('L', 'R'),
		(2, 2): ('L', 'R', 'U'),
		(2, 3): ('L', 'U'),
	}
	grid.set(rewards, actions)
	return grid

def print_values(V, grid):
	for i in range(grid.rows):
		print("---------------------------")
		for j in range(grid.cols):
			value = V.get((i, j), 0)
			if value >= 0:
				print("%.2f | " % value, end = "")
			else:
				print("%.2f | " % value, end = "") # -ve sign takes up an extra space
		print("")

def print_policy(P, grid):
	for i in range(grid.rows):
		print("---------------------------")
		for j in range(grid.cols):
			action = P.get((i, j), ' ')
			print("  %s  |" % action, end = "")
		print("")

if __name__ == "__main__":
    grid = standard_grid()

    print("\nreward : ")
    print_values(grid.rewards, grid)

    policy = initialize_random_policy(grid)
    
    print("\n초기 정책:")
    print_policy(policy, grid)

    V = {}
    states = grid.all_states()

    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0

    n_iter = 0
    MAX_ITER = 10000

    while(n_iter < MAX_ITER):
        maxChange = 0
        for s in states:
            oldValue = V[s]

            if s in policy:
                newValue = float('-inf')
                for a in ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)

                    v = r + GAMMA * V[grid.current_state()]

                    if v > newValue:
                        newValue = v

                V[s] = newValue
                maxChange = max(maxChange, np.abs(oldValue - V[s]))

        
        print("\n%i 번째 반복"% n_iter, end = '\n')
        print_values(V, grid)
        n_iter += 1

        if maxChange < DELTA_THRESHOLD:
            break


    for s in policy.keys():
        bestAction = None
        bestValue = float('-inf')

        for a in ACTIONS:
            grid.set_state(s)
            r = grid.move(a)

            v = r + GAMMA * V[grid.current_state()]

            if v > bestValue:
                bestValue = v
                bestAction = a

        policy[s] = bestAction

    print("\n가치 함수: ")
    print_values(V, grid)

    print("\n정책: ")
    print_policy(policy, grid)