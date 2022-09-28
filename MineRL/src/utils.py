import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from src.action import ActionShaping

def get_screen_compass(obs, device : str):
    screen = obs['pov'].transpose((2,0,1))
    _, screen_height, screen_width = screen.shape
    screen = np.ascontiguousarray(screen, dtype = np.float32)
    screen = torch.from_numpy(screen).unsqueeze(0).to(device)
    compass = torch.Tensor([obs['compass']['angle'].astype(np.float32).tolist() / 180.0]).unsqueeze(0).to(device)
    return screen, compass

def select_action(prob:torch.Tensor):
    m = Categorical(prob)
    a = m.sample().item()
    return a

def play(env, network : nn.Module, action_decision : ActionShaping, seed_num : int = 42, device :str = "cpu"):

    env.seed(seed_num)

    state = env.reset()
    state, compass = get_screen_compass(state, device)

    done = False
    sum_reward = 0

    network.eval()

    while not done:
        with torch.no_grad():
            state = state.to(device)
            compass = compass.to(device)
            prob, value = network(state, compass)
            action = select_action(prob)
            action_command = action_decision.action(action)

            next_state, reward, done, _ = env.step(action_command)
            sum_reward += reward
            reward = torch.tensor([reward], device = device)

            if not done:
                next_state, next_compass = get_screen_compass(next_state, device)
            else:
                next_state = None
                next_compass = None
            
            state = next_state
            compass = next_compass

            if done:
                break
            
    env.close()
    print("minerl play completed")

    return sum_reward