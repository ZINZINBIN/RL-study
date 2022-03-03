import gym
import random
import math
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

parser = argparse.ArgumentParser(description="training atari with Noise DQN")
parser.add_argument("--batch_size", type = int, default = 64)
parser.add_argument("--memory_size", type = int, default = 1000000)
parser.add_argument("--lr", type = float, default = 1e-1)
parser.add_argument("--gamma", type = float, default = 0.999)
parser.add_argument("--eps_start", type = float, default = 0.9)
parser.add_argument("--eps_end", type = float, default = 0.05)
parser.add_argument("--eps_decay", type = float, default = 200)
parser.add_argument("--target_update", type = int, default = 8)
parser.add_argument("--gpu_num", type = int, default = 0)
parser.add_argument("--num_episode", type = int, default = 128)
parser.add_argument("--wandb_save_name", type = str, default = "NoiseDQN-exp002")
parser.add_argument("--prob_alpha", type = float, default = 0.4)
parser.add_argument("--beta_start", type = float, default = 0.4)
parser.add_argument("--beta_frames", type = int, default = 1000)

args = vars(parser.parse_args())

BATCH_SIZE = args['batch_size']
GAMMA = args['gamma']
EPS_START = args['eps_start']
EPS_END = args['eps_end']
EPS_DECAY = args['eps_decay']
TARGET_UPDATE = args['target_update']
num_episode = args['num_episode']
memory_size = args['memory_size']
lr = args["lr"]
prob_alpha = args['prob_alpha']

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

# functino for use
def select_action(state, policy_net, device):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device = device, dtype = torch.long)

# initialize screen
# setting for centering the cart-pole and shape
init_screen = get_screen(env)
_,_,screen_height, screen_width = init_screen.shape

# Network loaded
policy_net = NoiseDQN(screen_height, screen_width, n_actions)
target_net = NoiseDQN(screen_height, screen_width, n_actions)
target_net.load_state_dict(policy_net.state_dict())

# gpu allocation(device)
policy_net.to(device)
target_net.to(device)

# target_network training -> x
# policy network만 학습 -> target network는 이후 load_state_dict()을 통해 가중치를 받아온다
target_net.eval()

# opimizer and memory loaded
optimizer = torch.optim.AdamW(policy_net.parameters(), lr = lr)
memory = PrioritzedReplayMemory(memory_size, prob_alpha, Transition)
# memory = ReplayMemory(memory_size)

def optimize_model(beta = beta_start):
    if len(memory) < BATCH_SIZE:
        return None, None

    transitions, indices, weights = memory.sample(BATCH_SIZE, beta)
    batch = Transition(*zip(*transitions))

    # 최종 상태가 아닌 경우의 mask
    non_final_mask = torch.tensor(
        tuple(
            map(lambda s : s is not None, batch.next_state)
        ),
        device = device,
        dtype = torch.bool
    )

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Q(s_t, a) computation
    # tensor -> gather(axis = 1, action_batch) -> tensor에서 각 행별 인덱스에 대응되는 값 호출
    state_action_values = policy_net(state_batch).gather(1, action_batch) 
    
    next_state_values = torch.zeros(BATCH_SIZE, device = device)

    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss() # Huber Loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # prioritized replay memory
    # priority update
    prios = loss.detach().cpu().numpy()
    prios = prios * weights[non_final_mask.detach().cpu().numpy()]
    prios += 1e-6

    memory.update_priorities(indices, prios)

    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1,1) # gradient clipping 

    optimizer.step()

    return expected_state_action_values.detach().cpu().numpy(), loss.detach().cpu().numpy()

reward_list = []
loss_list = []
mean_reward_list = []
mean_loss_list = []

if __name__ == "__main__":

    # wandb initialized
    wandb.init(project="DQN-Atari", entity="zinzinbin")

    # wandb experiment name edit
    wandb.run.name = args["wandb_save_name"]

    # save run setting
    wandb.run.save()

    # wandb setting
    wandb.config = {
        "learning_rate": lr,
        "episode": num_episode,
        "batch_size": BATCH_SIZE,
        "memory_size":memory_size,
        "gamma":GAMMA,
        "eps_start":EPS_START,
        "eps_end":EPS_END,
        "eps_decay":EPS_DECAY,
        "target_update":TARGET_UPDATE,
        "optimizer":"AdamW"
    }

    # training process for each episode
    for i_episode in tqdm(range(num_episode)):
        env.reset()
        state = get_screen(env)

        sum_reward = []
        sum_loss = []

        for t in count():
            state = state.to(device)
            action = select_action(state, policy_net, device)
            _, reward, done, _ = env.step(action.item())

            reward = torch.tensor([reward], device = device)

            if not done:
                next_state = get_screen(env)
                beta = beta_by_frame(t)

            else:
                next_state = None
            
            # memory에 transition 저장
            memory.push(state, action, next_state, reward)

            state = next_state

            # policy_net -> optimize
            reward_new, loss_new = optimize_model(beta)


            if reward_new is not None:
                sum_loss.append(loss_new)
                sum_reward.append(reward_new)

            if done:
                episode_durations.append(t+1)

                mean_loss = np.mean(sum_loss)
                mean_reward = np.mean(sum_reward)

                sum_loss = np.sum(sum_loss)
                sum_reward = np.sum(sum_reward)
                break

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())  

        loss_list.append(sum_loss)
        reward_list.append(sum_reward) 

        mean_loss_list.append(mean_loss)
        mean_reward_list.append(mean_reward)

        wandb.log({
            "episode_duration":len(episode_durations),
            "mean_loss":mean_loss, 
            "sum_loss":sum_loss,
            "mean_reward":mean_reward,
            "sum_reward":sum_reward
        })

        # memory cache delete
        gc.collect()

        # torch cache delete
        torch.cuda.empty_cache()

    print("training policy network and target network done....!")

    env.close()

    # plot the result
    plt.subplot(2,2,(1,1))
    plt.plot(range(1, num_episode + 1), mean_loss_list, 'r--', label = 'mean loss')
    plt.xlabel("Episode")
    plt.ylabel("Loss(mean)")
    plt.ylim([0, 1.0])
    plt.legend()

    plt.subplot(2,2,(2,1))
    plt.plot(range(1, num_episode + 1), loss_list, 'r--', label = 'sum loss')
    plt.xlabel("Episode")
    plt.ylabel("Loss(sum)")
    plt.ylim([0, 1.0])
    plt.legend()

    plt.subplot(2,2,(1,2))
    plt.plot(range(1, num_episode + 1), mean_reward_list, 'b--', label = 'mean reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward(mean)")
    plt.legend()

    plt.subplot(2,2,(2,2))
    plt.plot(range(1, num_episode + 1), reward_list, 'b--', label = 'sum reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward(sum)")
    plt.legend()

    plt.savefig("./results/Noise_DQN_loss_reward_curve.png")

    weights_save_dir = "./weights/noise_dqn_best_" + args['wandb_save_name'].split("-")[1] + ".pt"

    # save model weights 
    torch.save(policy_net.state_dict(), weights_save_dir)