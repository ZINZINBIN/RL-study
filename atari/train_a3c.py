import gym
import gc
from torch import long
import torch
import matplotlib.pyplot as plt
import argparse
from itertools import count
from src.utility import *
from src.model import *
from src.buffer import PrioritzedReplayMemory, ReplayMemory, Transition
from src.multiprocessing_env import *
from src.policy import *
from pyvirtualdisplay import Display
from tqdm import tqdm

parser = argparse.ArgumentParser(description="training atari with A3C method")
parser.add_argument("--hidden_dims", type = int, default = 128)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--num_envs", type = int, default = 8)
parser.add_argument("--weight_decay", type = float, default = 0.99)
parser.add_argument("--max_grad_norm", type = float, default = 0.5)
parser.add_argument("--gamma", type = float, default = 0.95)
parser.add_argument("--n_steps", type = int, default = 5)
parser.add_argument("--gpu_num", type = int, default = 0)
parser.add_argument("--max_frame", type = int, default = 10000)
parser.add_argument("--wandb_save_name", type = str, default = "A3C-exp001")
parser.add_argument("--value_loss_coeff", type = float, default = 1.0)
parser.add_argument("--policy_loss_coeff", type = float, default = 0.4)
parser.add_argument("--entropy_loss_coeff", type = float, default = 0.001)


args = vars(parser.parse_args())

hidden_dims = args["hidden_dims"]
lr = args["lr"]
num_envs = args["num_envs"]
weight_decay = args["weight_decay"]
n_steps = args["n_steps"]
max_frame = args['max_frame']
gamma = args['gamma']
value_loss_coeff = args['value_loss_coeff']
policy_loss_coeff = args['policy_loss_coeff']
entropy_loss_coeff = args['entropy_loss_coeff']
max_grad_norm = args['max_grad_norm']

# Generate display
display = Display(visible=False, size = (400,300))
display.start()

# decide environment 
env_name = 'Breakout-v0'
# env_name = 'CartPole-v0'

# using single thread(test)
env = gym.make(env_name).unwrapped
env.reset()

# using multi-thread 
def make_env():
    def _thunk():
        env = gym.make(env_name).unwrapped
        return env
    
    return _thunk

def get_multi_screen(envs):
    screen = envs.render(mode = 'rgb_array').transpose((2,0,1))
    screen = np.ascontiguousarray(screen, dtype = np.float32)
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0)

from src.multiprocessing_env import SubprocVecEnv

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

n_actions = env.action_space.n

# cuda check
if torch.cuda.is_available():
    print("cuda available : ", torch.cuda.is_available())
    print("cuda device count : ", torch.cuda.device_count())
    device = "cuda:" + str(args["gpu_num"])
else:
    device = "cpu" 

# initialize screen
init_screen = get_screen(env)
_,_,screen_height, screen_width = init_screen.shape

# Network Loaded
a3c = ActorCritic(screen_height, screen_width, n_actions, hidden_dims)
a3c.to(device)

# model summary
sample_inputs = torch.zeros_like(init_screen, device=device)
a3c.summary(sample_inputs)

# Opimizer
optimizer = torch.optim.RMSprop(a3c.parameters(), lr = lr, weight_decay=weight_decay)

# compute returns
def compute_returns(next_value, rewards, masks, gamma = 0.99):
    R = next_value
    returns = []

    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R) # 0번째 자리에 R 값을 집어넣는다(삽입)

    return returns

# test environment
def test_env():
    env.reset()
    state = get_screen(env)
    done = False
    total_reward = 0
    a3c.eval()

    while not done:
        with torch.no_grad():
            state = state.to(device)
            value, dist = a3c(state)
            action = dist.sample()
            _, reward, done, _ = env.step(action.item())
            total_reward += reward
            next_state = get_screen(env)
            state = next_state

    return total_reward

train_losses = []
test_rewards = []
test_frames = []

if __name__ == "__main__":

    # training process for each episode
    for frame_idx in tqdm(range(max_frame)):
        envs.reset()
        state = envs.render()

        log_probs = []
        rewards = []
        masks = []
        values = []
        losses = []
        entropy = 0

        # multi-steps  A3C algorithm : to avoid strong correlation between samples
        for n_step in range(n_steps):

            state = state.to(device)
            value, dist = a3c(state)
            action = dist.sample()
            _, reward, done, _ = envs.step(action.cpu().numpy())

            reward = torch.from_numpy(reward).to(device)
            state = envs.render()

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward.unsqueeze(1))
            masks.append(torch.FloatTensor(1-done).unsqueeze(1).to(device))

        # test plot
        if frame_idx % 100 == 0:
            test_reward = test_env()
            print("frame_idx : {}, test_reward : {:.3f}".format(frame_idx, test_reward))
            test_rewards.append(test_reward)
            test_frames.append(frame_idx)

        # next_state에 대한 예측값 
        log_probs = torch.cat(log_probs).to(device)
        next_state = state.to(device)
        next_value, next_dist = a3c(next_state)
    
        returns = compute_returns(next_value, rewards, masks, gamma)

        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()

        loss = policy_loss_coeff * policy_loss + value_loss_coeff * value_loss - entropy_loss_coeff * entropy
        optimizer.zero_grad()
        
        # gradient clipping 
        torch.nn.utils.clip_grad_norm_(a3c.parameters(), max_grad_norm)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        
        # memory cache delete
        gc.collect()

        # torch cache delete
        torch.cuda.empty_cache()

    print("training policy network and target network done....!")

    env.close()

    # plot the result
    plt.subplot(1,2,1)
    plt.plot(range(1, max_frame + 1), train_losses, 'r--', label = 'train loss')
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.ylim([0, 1.0])
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(test_frames, test_rewards, 'bo--', label = 'test reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.savefig("./results/A3C_loss_reward_curve.png")

    weights_save_dir = "./weights/a3c_" + args['wandb_save_name'].split("-")[1] + ".pt"

    # save model weights 
    torch.save(a3c.state_dict(), weights_save_dir)