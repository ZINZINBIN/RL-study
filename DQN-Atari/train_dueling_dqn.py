import gym
import random
import math
from torch import long
import torch
import matplotlib.pyplot as plt
import argparse
from itertools import count
from src.utility import *
from src.model import *
from src.buffer import ReplayMemory, Transition, PrioritzedReplayMemory
from pyvirtualdisplay import Display
from tqdm import tqdm

parser = argparse.ArgumentParser(description="training atari with Dueling DQN")
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--gamma", type = float, default = 0.999)
parser.add_argument("--eps_start", type = float, default = 0.9)
parser.add_argument("--eps_end", type = float, default = 0.05)
parser.add_argument("--eps_decay", type = float, default = 200)
parser.add_argument("--target_update", type = int, default = 10)
parser.add_argument("--num_episode", type = int, default = 256)
parser.add_argument("--prob_alpha", type = float, default = 0.6)
parser.add_argument("--beta", type = float, default = 0.4)

args = vars(parser.parse_args())

BATCH_SIZE = args['batch_size']
GAMMA = args['gamma']
EPS_START = args['eps_start']
EPS_END = args['eps_end']
EPS_DECAY = args['eps_decay']
TARGET_UPDATE = args['target_update']
num_episode = args['num_episode']
prob_alpha = args['prob_alpha']
beta = args['beta']

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
    device = "cuda:0"
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
policy_net = DuelingDQN(screen_height, screen_width, n_actions, n_actions, 128)
target_net = DuelingDQN(screen_height, screen_width, n_actions, n_actions, 128)
target_net.load_state_dict(policy_net.state_dict())

# gpu allocation(device)
policy_net.to(device)
target_net.to(device)

# target_network training -> x
# policy network만 학습 -> target network는 이후 load_state_dict()을 통해 가중치를 받아온다
target_net.eval()

# opimizer and memory loaded
optimizer = torch.optim.RMSprop(policy_net.parameters())
memory = PrioritzedReplayMemory(1000000, prob_alpha)

# 학습 루프 : DDQN의 경우 action evaluation 과 action selection을 분리함
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None, None
    transitions, indices, weights = memory.sample(BATCH_SIZE, beta)
    batch = Transition(*zip(*transitions))

    # 최종 상태가 아닌 경우의 mask : batch sample에 선택된 각각의 state에 대한 next state 여부
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
    # gather : Q(s,:) <- action 값에 따른 Q(s,a)를 구하기 위해 적용

    # DDQN과 DQN의 차이점 발생
    # Q(s,a) = R + gamma * Q(s_next, argmax(Q(s_next,a,w)), w_)
    # Pollicy Network에서 action selection 결정
    # Target Network에서 selected action에 따른 action evaluation 진행

    # Q(s_t+1, a_t+1) for target and policy network
    next_q_values = policy_net(non_final_next_states)
    next_q_state_values = target_net(non_final_next_states)

    # action that maximize Q(s_t+1, a_t+1) : torch.max(next_q_values, 1)[1].unsqueeze(1)
    # Q(s_t+1, argmax(Q(s_t+1, a, w)), w_)
    next_q_values_ddqn = torch.zeros(BATCH_SIZE, device = device)
    next_q_values_ddqn[non_final_mask] = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

    # Q(s_t, a_t)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Y for expected state action values
    expected_state_action_values = (next_q_values_ddqn * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss() # Huber Loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    prios = loss.detach().cpu().numpy()
    prios = prios * weights[non_final_mask.detach().cpu().numpy()]
    prios += 1e-5

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1,1) # gradient clipping 
    optimizer.step()

    memory.update_priorities(indices, prios)

    return expected_state_action_values.detach().cpu().numpy(), loss.detach().cpu().numpy()

mean_reward_list = []
mean_loss_list = []

# training process for each episode
for i_episode in tqdm(range(num_episode)):
    env.reset()
    state = get_screen(env)

    mean_reward = []
    mean_loss = []

    for t in count():
        state = state.to(device)
        action = select_action(state, policy_net, device)
        _, reward, done, _ = env.step(action.item())

        reward = torch.tensor([reward], device = device)

        if not done:
            next_state = get_screen(env)

        else:
            next_state = None
        
        # memory에 transition 저장
        memory.push(state, action, next_state, reward, done)

        state = next_state

        # policy_net -> optimize
        reward_new, loss_new = optimize_model()

        if reward_new is not None:
            mean_loss.append(loss_new)
            mean_reward.append(reward_new)

        if done:
            episode_durations.append(t+1)
            mean_loss = np.mean(mean_loss)
            mean_reward = np.mean(mean_reward)
            break

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())  

    mean_loss_list.append(mean_loss)
    mean_reward_list.append(mean_reward)  

print("training policy network and target network done....!")

env.close()

# plot the result
plt.subplot(1,2,1)
plt.plot(range(1, num_episode + 1), mean_loss_list, 'r--', label = 'mean loss')
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.ylim([0, 1.0])
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, num_episode + 1), mean_reward_list, 'b--', label = 'mean reward')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()

plt.savefig("./results/Dueling_DQN_loss_reward_curve.png")

# save model weights 
torch.save(policy_net.state_dict(), "./weights/dueling_dqn_best.pt")