from pyvirtualdisplay import Display
import matplotlib.pyplot as plt
import random
import gym

display = Display(visible= False, size = (400,300))
display.start()

episodes = 10

env = gym.make('CartPole-v0').unwrapped

for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = random.choice([0,1])
        n_state, reward, done, info = env.step(action)
        score += reward

    print("Episode : {}, Score : {}".format(episode, score))