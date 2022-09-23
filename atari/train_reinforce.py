import torch
import gym
from src.utility import get_screen
from src.model import PolicyNetwork
from src.reinforce import REINFORCE,Buffer
from pyvirtualdisplay import Display

# cuda check
if torch.cuda.is_available():
    print("cuda available : ", torch.cuda.is_available())
    print("cuda device count : ", torch.cuda.device_count())
    device = "cuda:1" 
else:
    device = "cpu"

if __name__  == "__main__":

    display = Display(visible=False, size = (400,300))
    display.start()

    env = gym.make('Breakout-v0').unwrapped
    env.reset()

    init_screen = get_screen(env)
    _,_,screen_height, screen_width = init_screen.shape

    print("h : ", screen_height)
    print("w : ", screen_width)

    n_actions = env.action_space.n
    hidden = 128

    policy = PolicyNetwork(screen_height, screen_width, hidden, n_actions)
    policy.to(device)

    buffer = Buffer(capacity = 100000)
    optimizer = torch.optim.AdamW(policy.parameters(), lr = 1e-3)
    gamma = 0.99
    num_episode = 1024
    verbose = 8

    REINFORCE(buffer, env,  policy, optimizer,  device, gamma, num_episode, verbose)