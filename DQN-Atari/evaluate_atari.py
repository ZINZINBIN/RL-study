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

if __name__ == "__main__":

    display = Display(visible=False, size = (400,300))
    display.start()

    env = gym.make(args['game']).unwrapped
    env.reset()

    state = get_screen(env)

    _,_,screen_height, screen_width = state.shape
    n_actions = env.action_space.n

    env.close()

    # model loaded
    policy_net = DQN(screen_height, screen_width, n_actions)
    policy_net.load_state_dict(torch.load(args['weight_dir'], map_location=device))
    policy_net.to(device)
    policy_net.eval()

    test_model(
        policy_net,
        device,
        args['game'],
        is_visible=False,
        test_episode=128,
        max_iter_per_episode=100000,
    )