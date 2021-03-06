import gym
import random
import math
import torch
import matplotlib.pyplot as plt
from itertools import count
from src.utility import *
from src.model import *
from src.buffer import ReplayMemory, Transition
from pyvirtualdisplay import Display
from tqdm import tqdm

# server 환경에서는 display를  직접 보여줄 수 없으므로 visible = false로 처리
display = Display(visible= False, size = (400,300))
display.start()

env = gym.make('CartPole-v0').unwrapped
env.reset()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
steps_done = 0
num_episode = 256

episode_durations = []

if torch.cuda.is_available():
    print("cuda available : ", torch.cuda.is_available())
    print("cuda device count : ", torch.cuda.device_count())
    device = "cuda:0"
else:
    device = "cpu" 

# epsilon - greedy algorithm으로 action 선택
def select_action(state, policy_net, device):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.* steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device = device, dtype = torch.long)

init_screen = get_screen(env)
_,_,screen_height, screen_width = init_screen.shape

print("input data shape : ", init_screen.shape)
print("screen_height : ", screen_height)
print("screen_width : ", screen_width)

# gym action space에서 action 상태 수 결정
n_actions = env.action_space.n

print("n_actions : ", n_actions)

policy_net = DQN(screen_height, screen_width, n_actions)
target_net = DQN(screen_height, screen_width, n_actions)
target_net.load_state_dict(policy_net.state_dict())

# model structure 
plot_model_struture(policy_net, init_screen.shape)

# device
policy_net.to(device)
target_net.to(device)

# target_net =>  eval
target_net.eval()

optimizer = torch.optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000000)

# 학습 루프
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None, None
    transitions = memory.sample(BATCH_SIZE)
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

    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1,1) # gradient clipping 

    optimizer.step()

    return expected_state_action_values.detach().cpu().numpy(), loss.detach().cpu().numpy()

mean_reward_list = []
mean_loss_list = []

for i_episode in tqdm(range(num_episode)):
    env.reset()
    last_screen = get_screen(env)
    current_screen = get_screen(env)

    state = current_screen - last_screen

    mean_reward = []
    mean_loss = []

    for t in count():
        state = state.to(device)
        action = select_action(state, policy_net, device)
        _, reward, done, _ = env.step(action.item())

        reward = torch.tensor([reward], device = device)
        last_screen = current_screen
        current_screen = get_screen(env)

        if not done:
            next_state = current_screen - last_screen

        else:
            next_state = None
        
        # memory에 transition 저장
        memory.push(state, action, next_state, reward)

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

plt.savefig("./results/DQN_loss_reward_curve.png")