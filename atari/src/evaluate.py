import torch
import torch.nn as nn
import numpy as np
import gym
from src.utility import EPS_DECAY_DEFAULT, EPS_END_DEFAULT, EPS_START_DEFAULT, get_screen, select_action_from_Q_Network
from typing import Optional
from tqdm import tqdm
from pyvirtualdisplay import Display

def test_model(
    policy_net, 
    device : Optional[str] = None,
    game:str = 'Breakout-v0',
    is_visible:bool = False,
    test_episode : int = 128,
    max_iter_per_episode : int = 10000,
    eps_start = EPS_START_DEFAULT,
    eps_end = EPS_END_DEFAULT,
    eps_decay = EPS_DECAY_DEFAULT
    ):

    display = Display(visible=is_visible, size = (400,300))
    display.start()

    env = gym.make(game).unwrapped
    # env = gym.wrappers.Monitor(env, "./results/play.avi",force=True)
    env.reset()

    n_actions = env.action_space.n

    n_iter = 0
    max_iters = max_iter_per_episode
    episode_reward_list = []
    
    # model loaded
    if device is None:
        device = 'cpu'

    policy_net.to(device)
    policy_net.eval()

    for episode in tqdm(range(test_episode), desc = "evaluation process", total = test_episode):
        env.reset()
        state = get_screen(env)
        steps_done = 0
        episode_reward = 0

        while True:
            state = state.to(device)
            action = select_action_from_Q_Network(
                state,
                policy_net,
                n_actions,
                steps_done,
                device,
                eps_start,
                eps_end,
                eps_decay
            )

            _, reward, done, _ = env.step(action)
            episode_reward += reward

            # update state from screen
            if not done:
                next_state = get_screen(env)

            else:
                next_state = None
            
            state = next_state

            # update iteration num
            n_iter +=1
            steps_done +=1

            if done:
                # print('episode : {}, Reward: {}'.format(episode, episode_reward))
                break
                
            if n_iter >= max_iters:
                # print('episode : {}, Reward: {}'.format(episode, episode_reward))
                break

        episode_reward_list.append(episode_reward)

    env.close()

    max_reward = max(episode_reward_list)
    mean_reward = sum(episode_reward_list) / len(episode_reward_list)

    print("# max reward : {:.2f} and mean reward : {:.2f}".format(max_reward, mean_reward))

    return episode_reward_list, max_reward, mean_reward


def play_model(
    policy_net : nn.Module, 
    device : Optional[str] = None,
    game:str = 'Breakout-v0',
    is_visible:bool = False,
    ):

    display = Display(visible=is_visible, size = (400,300))
    display.start()

    env = gym.make(game)
    env = gym.wrappers.Monitor(env, "./results/play.avi",force=True)
    env.reset()

    n_actions = env.action_space.n

    # model loaded
    if device is None:
        device = 'cpu'

    policy_net.to(device)
    policy_net.eval()

    state = get_screen(env)
    steps_done = 0
    episode_reward = 0

    while True:
        state = state.to(device)
        action = select_action_from_Q_Network(
            state,
            policy_net,
            n_actions,
            steps_done,
            device
        )

        _, reward, done, _ = env.step(action)
        episode_reward += reward

        # update state from screen
        if not done:
            next_state = get_screen(env)

        else:
            next_state = None
        
        state = next_state

        # update iteration num
        steps_done +=1

        if done:
            break
            
    env.close()

    print("# reward : {:.2f}".format(episode_reward))

if __name__ == "__main__":

    if torch.cuda.is_available():
        print("cuda available : ", torch.cuda.is_available())
        print("cuda device count : ", torch.cuda.device_count())
        device = "cuda:0"
    else:
        device = "cpu" 

    from src.model import DuelingDQN

    display = Display(visible=False, size = (400,300))
    display.start()

    env = gym.make('Breakout-v0').unwrapped
    # env = gym.make("PongNoFrameskip-v4").unwrapped
    env.reset()

    # generate screen and state
    state = get_screen(env)

    _,_,screen_height, screen_width = state.shape
    n_actions = env.action_space.n
    env.close()

    policy_net = DuelingDQN(screen_height, screen_width, n_actions, n_actions, 128)
    policy_net.load_state_dict('./weights/dueling_dqn_best.pt', map_location=device)
    policy_net.to(device)
    policy_net.eval()    

    play_model(policy_net, device, 'Breakout-v0', False)
    