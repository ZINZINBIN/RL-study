import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import minerl
import torch
import logging
from pyvirtualdisplay import Display
from src.model import PPO
from src.buffer import ReplayBuffer
from src.utils import play

# parameter
gamma = 0.99
num_episode = 128
hidden_dims = 256
lr = 1e-3
n_actions = 7
seed_num = 42
eps_clip = 0.1
lamda = 0.9
entropy_coeff = 0.1
T_horizon = 128
k_epoch = 4

camera_angle = 20
always_attack = True

episode_durations = []
steps_done = 0

screen_height = 64
screen_width = 64

# torch cuda setting
if torch.cuda.is_available():
    print("cuda available : ", torch.cuda.is_available())
    print("cuda device count : ", torch.cuda.device_count())
    device = "cuda:0"
else:
    device = "cpu" 

if __name__ =="__main__":
    display = Display(visible=False, size=(400, 300))
    display.start();

    # logger
    logging.basicConfig(level=logging.DEBUG)

    env = gym.make('MineRLNavigateDense-v0')

    env = VideoRecorder(env, './video')

    # model
    network = PPO(screen_height, screen_width, n_actions, 1, hidden_dims, 1).to(device)
    network.load_state_dict(torch.load("./weights/ppo_best.pt"))
    network.eval()
                                
    total_reward = play(env, network, seed_num = 42, device = device)

    env.release()
    env.play()

    print(f'\nTotal reward = {total_reward}')