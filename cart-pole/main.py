import matplotlib
import gym
import matplotlib
import matplotlib.pyplot as plt
from src.utility import *

if __name__ == '__main__':
    env = gym.make('CartPole-v0').unwrapped

    is_python = 'inline' in matplotlib.get_backend()

    if is_python:
        from IPython import display

    plt.ion()

    env.reset()
    plt.figure()
    plt.imshow(get_screen(env).cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
    plt.title('Example extracted screen')
    plt.show()