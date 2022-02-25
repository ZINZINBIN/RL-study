import gym
import numpy as np
import os
from pyvirtualdisplay import Display
from src.model import *
from src.utility import *

# torch initialize
if torch.cuda.is_available():
    print("cuda available : ", torch.cuda.is_available())
    print("cuda device count : ", torch.cuda.device_count())
    device = "cuda:0"
else:
    device = "cpu" 

if __name__ == "__main__":
    display = Display(visible=False, size = (400,300))
    display.start()

    env = gym.make('Breakout-v0').unwrapped
    env.reset()

    init_screen = get_screen(env)
    _,_,screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n

    n_iter = 0
    max_iters = 100000
    episode_reward = 0
    steps_done = 0

    # model loaded
    policy_net = DQN(screen_height, screen_width, n_actions)
    policy_net.load_state_dict(torch.load("./weights/ddqn_best.pt", map_location=device))
    policy_net.to(device)


    while True:

        n_iter +=1
        steps_done +=1

        # generate screen and state
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen
        
        state = state.to(device)
        action = select_action_from_Q_Network(
            state,
            policy_net,
            n_actions,
            steps_done,
            device,
        )

        _, reward, done, _ = env.step(action)
        episode_reward += reward

        if done:
            print('Reward: %s' % episode_reward)
            break
            
        if n_iter >= max_iters:
            break

    env.close()