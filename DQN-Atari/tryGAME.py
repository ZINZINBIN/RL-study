import gym
import numpy as np
import argparse
import os
from pyvirtualdisplay import Display
from src.model import *
from src.utility import *
from src.evaluate import *

parser = argparse.ArgumentParser(description="test model")
parser.add_argument("--weight_dir", type = str, default = "./weights/dqn_best_exp001.pt")
parser.add_argument("--game", type = str, default = 'Breakout-v0')
args = vars(parser.parse_args())

# torch initialize
if torch.cuda.is_available():
    print("cuda available : ", torch.cuda.is_available())
    print("cuda device count : ", torch.cuda.device_count())
    device = "cuda:0"
else:
    device = "cpu" 

def play_model(
    policy_net : nn.Module, 
    device : Optional[str] = None,
    game:str = 'Breakout-v0',
    is_visible:bool = False,
    ):

    display = Display(visible=is_visible, size = (400,300))
    display.start()

    env = gym.make(game)
    env = gym.wrappers.Monitor(env, "./results/play.avi", force=True)
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
    policy_net.load_state_dict(torch.load('./weights/dueling_dqn_best.pt', map_location=device))
    policy_net.to(device)
    policy_net.eval()    

    play_model(policy_net, device, 'Breakout-v0', False)
    

'''
if __name__ == "__main__":
    display = Display(visible=False, size = (400,300))
    display.start()

    env = gym.make(args['game']).unwrapped
    # env = gym.make("PongNoFrameskip-v4").unwrapped
    env.reset()

    # generate screen and state
    state = get_screen(env)

    _,_,screen_height, screen_width = state.shape
    n_actions = env.action_space.n

    n_iter = 0
    max_iters = 10000
    episode_reward = 0
    steps_done = 0

    # model loaded
    policy_net = DQN(screen_height, screen_width, n_actions)
    policy_net.load_state_dict(torch.load(args['weight_dir'], map_location=device))
    policy_net.to(device)
    policy_net.eval()

    while True:
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
            print('Reward: %s' % episode_reward)
            break
            
        if n_iter >= max_iters:
            print('Reward: %s' % episode_reward)
            break

    env.close()
'''