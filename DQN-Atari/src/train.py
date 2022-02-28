import torch
import torch.nn as nn
import numpy as np
from src.buffer import ReplayMemory, Transition
from src.utility import EPS_DECAY_DEFAULT, EPS_END_DEFAULT, EPS_START_DEFAULT, get_screen, select_action_from_Q_Network
from typing import Optional
from tqdm import tqdm
from itertools import count
import wandb
import gc

def optimize_dqn(
    memory, 
    policy_net, 
    target_net, 
    optimizer  = None,
    criterion = None,
    BATCH_SIZE : int = 128, 
    GAMMA : float = 0.99, 
    device : Optional[str] = "cpu"
    ):

    if len(memory) < BATCH_SIZE:
        return None, None

    if device is None:
        device = "cpu"
    
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

    if optimizer is None:
        optimizer = torch.optim.RMSprop(policy_net.parameters(), lr = 1e-3)

    if criterion is None:
        criterion = nn.SmoothL1Loss() # Huber Loss

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1,1) # gradient clipping 

    optimizer.step()

    return expected_state_action_values.detach().cpu().numpy(), loss.detach().cpu().numpy()

def train_dqn(
    env, 
    policy_net, 
    target_net,
    memory,
    optimizer = None,
    criterion = None,
    TARGET_UPDATE : int = 12,
    batch_size : int = 128,
    gamma : float = 0.99,
    num_episode : int = 128, 
    device : Optional[str] = "cpu",
    eps_start = EPS_START_DEFAULT,
    eps_end = EPS_END_DEFAULT,
    eps_decay = EPS_DECAY_DEFAULT,
    wandb_monitoring : bool = False
    ):

    if device is None:
        device = "cpu"
    
    steps_done = 0
    n_actions = env.action_space.n

    episode_durations = []
    mean_loss_list = []
    mean_reward_list = []

    for i_episode in tqdm(range(num_episode)):
        env.reset()
        state = get_screen(env)

        mean_reward = []
        mean_loss = []

        for t in count():
            state = state.to(device)
            action = select_action_from_Q_Network(state, policy_net, n_actions, steps_done, device, eps_start, eps_end, eps_decay)
            _, reward, done, _ = env.step(action.item())

            reward = torch.tensor([reward], device = device)

            if not done:
                next_state = get_screen(env)

            else:
                next_state = None
            
            # memory에 transition 저장
            memory.push(state, action, next_state, reward)

            state = next_state

            # policy_net -> optimize
            reward_new, loss_new = optimize_dqn(
                memory,
                policy_net,
                target_net,
                optimizer,
                criterion,
                batch_size,
                gamma,
                device
            )

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

        if wandb_monitoring:
            wandb.log({"mean_loss":mean_loss, "mean_reward":mean_reward})

        # memory cache delete
        gc.collect()

        # torch cache delete
        torch.cuda.empty_cache()

    print("training policy network and target network done....!")
    env.close()