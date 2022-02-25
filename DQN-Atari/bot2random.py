import gym
import random
from pyvirtualdisplay import Display

GAME_LIST = [
    'SpaceInvaders-v0',
    'Breakout-v0'
]

def main():
    display = Display(visible=False, size = (400,300))
    display.start()

    env = gym.make(GAME_LIST[1]).unwrapped
    env.reset()

    screen = env.render(mode = 'rgb_array').transpose((2,0,1))
    print("env.render screen shape : ",screen.shape)
    print("env n action : ", env.action_space.n)

    n_iter = 0
    max_iters = 10000

    episode_reward = 0

    while True:
        n_iter +=1
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        episode_reward += reward

        if done:
            print('Reward: %s' % episode_reward)
            break
            
        if n_iter >= max_iters:
            break

    env.close()
    
if __name__ == '__main__':
    main()