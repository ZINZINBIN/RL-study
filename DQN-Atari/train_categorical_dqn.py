from email import policy
import gym
import random
import math
from src.train import optimize_categorical_DQN
import wandb
import gc
from torch import long
import torch
import matplotlib.pyplot as plt
import argparse
from itertools import count
from src.utility import *
from src.model import *
from src.buffer import PrioritzedReplayMemory, ReplayMemory, Transition
from pyvirtualdisplay import Display
from tqdm import tqdm

parser = argparse.ArgumentParser(description="training atari with categorical DQN")
parser.add_argument("--batch_size", type = int, default = 64)
parser.add_argument("--memory_size", type = int, default = 100000)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--gamma", type = float, default = 0.999)
parser.add_argument("--target_update", type = int, default = 8)
parser.add_argument("--gpu_num", type = int, default = 1)
parser.add_argument("--num_episode", type = int, default = 64)
parser.add_argument("--wandb_save_name", type = str, default = "CategoricalDQN-exp001")
parser.add_argument("--prob_alpha", type = float, default = 0.4)
parser.add_argument("--beta_start", type = float, default = 0.4)
parser.add_argument("--beta_frames", type = int, default = 1000)
parser.add_argument("--num_atoms", type = int, default = 51)
parser.add_argument("--V_min", type = int, default = -10)
parser.add_argument("--V_max", type = int, default = 10)
parser.add_argument("--hidden_dims", type = int, default = 128)

args = vars(parser.parse_args())

BATCH_SIZE = args['batch_size']
GAMMA = args['gamma']
TARGET_UPDATE = args['target_update']
num_episode = args['num_episode']
memory_size = args['memory_size']
lr = args["lr"]
prob_alpha = args['prob_alpha']
num_atoms = args['num_atoms']
V_min = args['V_min']
V_max = args['V_max']
hidden_dims = args['hidden_dims']

# Prioritized Replay Memory beta 값은 frame에 따라 변화
beta_start = args['beta_start']
beta_frames = args['beta_frames']
beta_by_frame = lambda frame_idx : min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

episode_durations = []
steps_done = 0

display = Display(visible=False, size = (400,300))
display.start()

env = gym.make('Breakout-v0').unwrapped
env.reset()

n_actions = env.action_space.n

# cuda check
if torch.cuda.is_available():
    print("cuda available : ", torch.cuda.is_available())
    print("cuda device count : ", torch.cuda.device_count())
    device = "cuda:" + str(args["gpu_num"])
else:
    device = "cpu" 

# initialize screen
# setting for centering the cart-pole and shape
init_screen = get_screen(env)
_,_,screen_height, screen_width = init_screen.shape

# Network loaded
policy_net = CategoricalDQN(screen_height, screen_width, n_actions, num_atoms, V_min, V_max, hidden_dims)
target_net = CategoricalDQN(screen_height, screen_width, n_actions, num_atoms, V_min, V_max, hidden_dims)
target_net.load_state_dict(policy_net.state_dict())

# gpu allocation(device)
policy_net.to(device)
target_net.to(device)

# target_network training -> x
# policy network만 학습 -> target network는 이후 load_state_dict()을 통해 가중치를 받아온다
policy_net.train()
target_net.eval()

# opimizer and memory loaded
optimizer = torch.optim.AdamW(policy_net.parameters(), lr = lr)
memory = ReplayMemory(memory_size)

reward_list = []
loss_list = []
mean_reward_list = []
mean_loss_list = []

if __name__ == "__main__":

    # # wandb initialized
    # wandb.init(project="DQN-Atari", entity="zinzinbin")

    # # wandb experiment name edit
    # wandb.run.name = args["wandb_save_name"]

    # # save run setting
    # wandb.run.save()

    # # wandb setting
    # wandb.config = {
    #     "learning_rate": lr,
    #     "episode": num_episode,
    #     "batch_size": BATCH_SIZE,
    #     "memory_size":memory_size,
    #     "gamma":GAMMA,
    #     "num_atoms":num_atoms,
    #     "hidden_dims":hidden_dims,
    #     "V_min":V_min,
    #     "V_max":V_max,
    #     "target_update":TARGET_UPDATE,
    #     "optimizer":"AdamW"
    # }

    # training process for each episode
    for i_episode in tqdm(range(num_episode)):
        env.reset()
        state = get_screen(env)

        sum_reward = []
        sum_loss = []

        for t in count():
            state = state.to(device)
            action = policy_net.act(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward])
            done = torch.tensor([done], dtype = torch.int32)

            if not done:
                next_state = get_screen(env)
                beta = beta_by_frame(t)
            else:
                next_state_memory = None
            
            # memory에 transition 저장
            memory.push(state.cpu(), action.cpu(), next_state.cpu(), reward.cpu(), done.cpu())
            state = next_state

            # policy_net -> optimize
            loss = optimize_categorical_DQN(
                memory,
                target_net,
                policy_net,
                optimizer, 
                beta,
                GAMMA,
                BATCH_SIZE,
                device
            )

            try:
                loss = loss.cpu().item()
                reward = reward.cpu().item()
                sum_loss.append(loss)
                sum_reward.append(reward)
            except:
                loss = 0
                reward = 0
                sum_loss.append(loss)
                sum_reward.append(reward)

            if done.item():
                episode_durations.append(t+1)

                mean_loss = np.mean(sum_loss)
                mean_reward = np.mean(sum_reward)

                sum_loss = np.sum(sum_loss)
                sum_reward = np.sum(sum_reward)

                print("mean_loss : {:.3f}, mean_sum : {:.3f}, sum_loss : {:.3f}, sum_reward:{:.3f}".format(mean_loss, mean_reward, sum_loss, sum_reward))
                break

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())  

        loss_list.append(sum_loss)
        reward_list.append(sum_reward) 

        mean_loss_list.append(mean_loss)
        mean_reward_list.append(mean_reward)

        # wandb.log({
        #     "episode_duration":len(episode_durations),
        #     "mean_loss":mean_loss, 
        #     "sum_loss":sum_loss,
        #     "mean_reward":mean_reward,
        #     "sum_reward":sum_reward
        # })

        # memory cache delete
        gc.collect()

        # torch cache delete
        torch.cuda.empty_cache()

    print("training policy network and target network done....!")

    env.close()

    # plot the result
    plt.figure(figsize = (12,12))
    plt.subplot(2,2,1)
    plt.plot(range(1, num_episode + 1), mean_loss_list, 'r--', label = 'mean loss')
    plt.xlabel("Episode")
    plt.ylabel("Loss(mean)")
    plt.ylim([0, 1.0])
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(range(1, num_episode + 1), loss_list, 'r--', label = 'sum loss')
    plt.xlabel("Episode")
    plt.ylabel("Loss(sum)")
    plt.ylim([0, 1.0])
    plt.legend()

    plt.subplot(2,2,3)
    plt.plot(range(1, num_episode + 1), mean_reward_list, 'b--', label = 'mean reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward(mean)")
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(range(1, num_episode + 1), reward_list, 'b--', label = 'sum reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward(sum)")
    plt.legend()

    plt.savefig("./results/Categorical_DQN_loss_reward_curve.png")

    weights_save_dir = "./weights/categorical_dqn_best_" + args['wandb_save_name'].split("-")[1] + ".pt"

    # save model weights 
    torch.save(policy_net.state_dict(), weights_save_dir)