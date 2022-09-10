import os
import argparse
import gym
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display
from src.policy_gradient import *
from src.buffer import *

parser = argparse.ArgumentParser(description="training cart pole with policy gradient method")
parser.add_argument("--episodes", type = int, default = 128)
parser.add_argument("--num_traj", type = int, default = 16)
parser.add_argument("--max_num_steps", type = int, default = 1024)
parser.add_argument("--gamma", type = float, default = 0.95)
parser.add_argument("--policy_learning_rate", type = float, default = 1e-3)
parser.add_argument("--value_learning_rate", type = float, default = 1e-3)
parser.add_argument("--policy_saved_path", type = str, default = "./weights/mcpg_policy.pt")
parser.add_argument("--value_saved_path", type = str, default = "./weights/mcpg_value.pt")
parser.add_argument("--epi_verbose", type = int, default = 10)

args = parser.parse_args()

display = Display(visible=False, size = (400,300))
display.start()

env = gym.make('CartPole-v0').unwrapped
env.reset()

# cuda check
if torch.cuda.is_available():
    print("cuda available : ", torch.cuda.is_available())
    print("cuda device count : ", torch.cuda.device_count())
    device = "cuda:0"
else:
    device = "cpu" 

if __name__ == "__main__":
    policy_net, value_net, mean_returns_list = policy_gradient_process(
        env,
        args.episodes,
        args.num_traj,
        args.max_num_steps,
        args.gamma,
        args.policy_learning_rate,
        args.value_learning_rate,
        args.policy_saved_path,
        args.value_saved_path,
        device,
        args.epi_verbose
    )
    epis = range(1, len(mean_returns_list) + 1)
    plt.figure(1)
    plt.plot(epis, mean_returns_list, 'ro--', label = "mean returns")
    plt.xlabel("Episode")
    plt.ylabel("mean return")
    plt.title("Mean Returns curve via policy gradient training")
    plt.savefig("./results/mean-returns-curve-policy-gradient.png")
    