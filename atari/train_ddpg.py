import torch
import gym
import matplotlib.pyplot as plt
from src.utility import get_screen
from src.ddpg import train_ddpg, ActorNetwork, CriticNetwork, ReplayBuffer, NormalizedActions
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

    env = NormalizedActions(gym.make('Pendulum-v1'))
    env.reset()

    init_screen = get_screen(env)
    _,_,screen_height, screen_width = init_screen.shape

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden = 128
    replay_size = 100000
    lr = 1e-3
    tau = 1e-2
    gamma = 0.99
    num_episode = 256
    verbose = 8
    batch_size = 64
    min_value = -1.0
    max_value = 1.0
    verbose = 8
    
    policy_network = ActorNetwork(screen_height,screen_width,action_dim, hidden)
    target_policy_network = ActorNetwork(screen_height,screen_width,action_dim, hidden)

    value_network = CriticNetwork(screen_height, screen_width, action_dim, hidden)
    target_value_network = CriticNetwork(screen_height, screen_width, action_dim, hidden)
    
    policy_network.to(device)
    target_policy_network.to(device)

    value_network.to(device)
    target_value_network.to(device)

    memory = ReplayBuffer(replay_size)

    value_optimizer = torch.optim.AdamW(value_network.parameters(), lr = lr)
    policy_optimizer = torch.optim.AdamW(policy_network.parameters(), lr = lr)

    value_loss_fn = torch.nn.SmoothL1Loss(reduction = 'mean')
    
    episode_durations, episode_rewards = train_ddpg(
        env, 
        memory,
        policy_network,
        value_network,
        target_policy_network,
        target_value_network,
        policy_optimizer,
        value_optimizer,
        value_loss_fn,
        batch_size,
        gamma,
        device,
        min_value,
        max_value,
        tau,
        num_episode,
        verbose
    )

    plt.subplot(1,2,1)
    plt.plot(range(1, num_episode + 1), episode_durations, 'r--', label = 'episode duration')
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1, num_episode + 1), episode_rewards, 'b--', label = 'episode reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.savefig("./results/DDPG_episode_reward.png")

    # evaluate
    