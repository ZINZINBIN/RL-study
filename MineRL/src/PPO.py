import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Optional, Union
from src.buffer import ReplayBuffer, Transition
from src.action import ActionShaping
from src.utils import get_screen_compass, select_action
from tqdm.auto import tqdm
import time, gc

# optimization : TD loss + clipping for PPO
def optimize(
    memory : ReplayBuffer,
    network : nn.Module,
    optimizer : torch.optim.Optimizer,
    criterion : nn.Module,
    gamma : float = 0.99,
    lamda : float = 0.1,
    eps_clip : float = 0.1,
    entropy_coeff : float = 0.1,
    device : Optional[str] = "cpu",
    k_epoch : int = 16
    ):
    
    if device is None:
        device = "cpu"

    if optimizer is None:
        optimizer = torch.optim.RMSprop(network.parameters(), lr = 1e-3)

    if criterion is None:
        criterion = nn.SmoothL1Loss() # Huber Loss
    
    transition = memory.get_trajectory()
    memory.clear()

    traj = Transition(*zip(*transition))

    non_final_mask = torch.tensor(
        tuple(
            map(lambda s : s is not None, traj.next_state)
        ),
        device = device,
        dtype = torch.bool
    )

    non_final_next_states = torch.cat([s for s in traj.next_state if s is not None]).to(device)
    non_final_next_compass = torch.cat([s for s in traj.next_compass if s is not None]).to(device)

    # state, action, reward as tensor
    state = torch.cat(traj.state).to(device)[non_final_mask]
    compass = torch.cat(traj.compass).to(device)[non_final_mask]
    action = torch.cat(traj.action).to(device)[non_final_mask]
    reward = torch.cat(traj.reward).to(device)[non_final_mask]
    prob_a = torch.cat(traj.prob_a).to(device)[non_final_mask]

    for epoch in range(k_epoch):

        next_log_prob, next_value = network(non_final_next_states, non_final_next_compass)
        pi, value = network(state, compass)

        td_target = reward + gamma * next_value
        delta = td_target - value
        delta = delta.detach().cpu().numpy()

        adv_list = []
        adv = 0

        for delta_t in delta[::-1]:
            adv = gamma * lamda * adv + delta_t[0]
            adv_list.append([adv])

        adv_list.reverse()
        adv = torch.FloatTensor(adv_list).to(device)

        pi_a = pi.gather(1, action)
        ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

        m = Categorical(pi)
        entropy = m.entropy().mean()
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv

        # print("surr1 : ", surr1.size())
        # print("surr2 : ", surr2.size())
        # print("value : ", value.size())
        # print("td_target : ", td_target.size())
        # print("entropy : ", entropy.size())

        loss = -torch.min(surr1, surr2) + criterion(value, td_target.detach()) - entropy_coeff * entropy

        optimizer.zero_grad()
        loss.mean().backward(retain_graph = True)

        for param in network.parameters():
            param.grad.data.clamp_(-1,1) # gradient clipping 

        optimizer.step()

    return loss

def train(
    env,
    memory : ReplayBuffer,
    network : nn.Module,
    optimizer : torch.optim.Optimizer,
    criterion : nn.Module,
    T_horizon : int,
    gamma : float = 0.99,
    lamda : float = 0.1,
    eps_clip : float = 0.1,
    entropy_coeff : float = 0.1,
    device : Optional[str] = "cpu",
    num_episode : int = 1024, 
    save_dir : str = "./weights/ppo_last.pt",
    camera_angle : int = 10,
    always_attack : bool = True,
    seed_num = 42,
    k_epoch : int = 16
    ):

    steps_done = 0

    episode_durations = []
    reward_list = []
    loss_list = []

    action_decision = ActionShaping(env, camera_angle = camera_angle, always_attack = always_attack)

    for i_episode in tqdm(range(num_episode)):
        env.seed(seed_num)
        state = env.reset()
        done = False
        state, compass = get_screen_compass(state, device)

        sum_reward = 0
        mean_loss = 0
        sum_loss = []

        start_time = time.time()

        for t in range(T_horizon):
            state = state.to(device)
            compass = compass.to(device)
            prob, value = network(state, compass)
            prob = prob.squeeze(0)
            action = select_action(prob)
            action_command = action_decision.action(action)
            next_state, reward, done, _ = env.step(action_command)

            prob_a = prob[action].item()

            sum_reward += reward
            reward = torch.tensor([reward], device = device)
            action = torch.tensor([action], device = device).view(1,1)
            prob_a = torch.tensor([prob_a], device = device)

            if not done:
                next_state, next_compass = get_screen_compass(next_state, device)
            else:
                next_state = None
                next_compass = None
            
            # memory에 transition 저장
            memory.push(state, compass, action, next_state, next_compass, reward, done, prob_a)

            state = next_state
            compass = next_compass

            if done:
                break
        
        end_time = time.time()
        dt = end_time - start_time

        loss_new = optimize(
            memory,
            network,
            optimizer,
            criterion,
            gamma,
            lamda,
            eps_clip,
            entropy_coeff,
            device,
            k_epoch
        )

        if loss_new is not None:
            loss_list.append(loss_new)

        reward_list.append(sum_reward) 

        print("i_eposide : {} and reward : {}, time spend : {}".format(i_episode, sum_reward, dt))

        # memory cache delete
        gc.collect()

        # torch cache delete
        torch.cuda.empty_cache()

        # model save
        torch.save(network.state_dict(), save_dir)

    print("training policy network and target network done....!")
    env.close()

    return loss_list, reward_list