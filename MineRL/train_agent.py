import gym
import minerl
import torch
import logging
from pyvirtualdisplay import Display
from src.model import PPO
from src.buffer import ReplayBuffer
from src.PPO import train

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
T_horizon = 1024 * 4
k_epoch = 4

camera_angle = 20
always_attack = True

episode_durations = []
steps_done = 0

screen_height = 64
screen_width = 64

logging.disable(logging.ERROR) # reduce clutter, remove if something doesn't work to see the error logs.
logging.disable(logging.WARNING) 

# torch cuda setting
if torch.cuda.is_available():
    print("cuda available : ", torch.cuda.is_available())
    print("cuda device count : ", torch.cuda.device_count())
    device = "cuda:0"
else:
    device = "cpu" 

if __name__ == "__main__":

    display = Display(visible=False, size=(400, 300))
    display.start();

    # logger
    # logging.basicConfig(level=logging.DEBUG)

    # define environment
    env = gym.make('MineRLNavigateDense-v0')

    # memory
    memory = ReplayBuffer(T_horizon)

    # model
    network = PPO(screen_height, screen_width, n_actions, 1, hidden_dims, 1)
    network.to(device)

    # optimizer
    optimizer = torch.optim.AdamW(network.parameters(), lr = lr)

    # loss_fn
    loss_fn = torch.nn.SmoothL1Loss(reduction = 'none')  

    # train process
    train(
        env,
        memory,
        network,
        optimizer,
        loss_fn,
        T_horizon,
        gamma,
        lamda,
        eps_clip,
        entropy_coeff,
        device,
        num_episode, 
        save_dir = "./weights/ppo_last.pt",
        camera_angle = camera_angle,
        always_attack = always_attack,
        seed_num = seed_num,
        k_epoch = k_epoch
    )