import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from pytorch_model_summary import summary

resize = T.Compose([
    T.ToPILImage(),
    T.Resize(128, interpolation=Image.CUBIC),
    T.ToTensor()
])

EPS_START_DEFAULT = 0.9
EPS_END_DEFAULT = 0.05
EPS_DECAY_DEFAULT = 200

def get_screen(env):
    # convert 800 * 1200 * 3 to 400 * 600 * 3
    screen = env.render(mode = 'rgb_array').transpose((2,0,1))
    _, screen_height, screen_width = screen.shape
    # screen = screen[:, int(screen_height * 0.4) : int(screen_height * 0.8)]
    # view_width = int(screen_width * 0.6)

    # continous한 memory 형태로 반환
    screen = np.ascontiguousarray(screen, dtype = np.float32)
    screen = torch.from_numpy(screen)

    return resize(screen).unsqueeze(0)

def select_action_from_Q_Network(
    state, 
    policy_net, 
    n_actions : int,
    steps_done : int,
    device = 'cpu', 
    eps_start = EPS_START_DEFAULT,
    eps_end = EPS_END_DEFAULT,
    eps_decay = EPS_DECAY_DEFAULT
    ):

    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device = device, dtype = torch.long)


def plot_durations(episode_duration):
    plt.figure(1)
    plt.clf()

    durations_t = torch.tensor(episode_duration, dtype = torch.float)

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')

    plt.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
        
def plot_model_struture(model, input_shape):
    x = torch.zeros(input_shape)
    # 앞에서 생성한 model에 Input을 x로 입력한 뒤 (model(x))  graph.png 로 이미지를 출력합니다.
    # make_dot(model(x), params=dict(model.named_parameters())).render("graph", format="png")
    print(summary(model, x, show_input = True))


# A Distributional Perspective on Reinforcement Learning
# projection distribution

def projection_distribution(target_network:torch.nn.Module, next_state, rewards, dones, device = 'cpu'):

    assert hasattr(target_network, 'V_max')
    assert hasattr(target_network, 'V_min')
    assert hasattr(target_network, 'num_atoms')

    V_max = target_network.V_max
    V_min = target_network.V_min
    num_atoms = target_network.num_atoms

    batch_size = next_state.size(0)

    delta_z = float(V_max - V_min) / (num_atoms - 1)
    support = torch.linspace(V_min, V_max, num_atoms)

    next_dist = target_network(next_state).data.cpu() * support
    next_action = next_dist.sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    next_dist = next_dist.gather(1, next_action).squeeze(1)

    rewards = rewards.unsqueeze(1).expand_as(next_dist).to(device)
    dones = dones.unsqueeze(1).expand_as(next_dist).to(device)
    support = support.unsqueeze(0).expand_as(next_dist).to(device)

    Tz = rewards + (1-dones) * 0.99 * support
    Tz = Tz.clamp(min=V_min, max = V_max).cpu()

    b = (Tz - V_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long().unsqueeze(1).expand(batch_size, num_atoms)
    proj_dist = torch.zeros(next_dist.size())
    proj_dist.view(-1).index_add_(0, (l+offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u+offset).view(-1), (next_dist * (b-l.float())).view(-1))

    return proj_dist

def projection_distribution_QR(target_network:torch.nn.Module, num_quant, dist, next_state, rewards, dones, device = 'cpu'):

    batch_size = next_state.size(0)
    
    next_dist = target_network(next_state) # (batch_size, n_actions, num_quant)
    next_action = next_dist.mean(2).max(1)[1] #(batch_size)
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_quant)

    # gather : next_dist에 대해서 dim = 1에 대해 next_action 값을 인덱스로 데이터를 가져온다.
    next_dist = next_dist.gather(1, next_action).squeeze(1).cpu().data # (batch_size, num_quant)..?

    expected_quant = rewards.unsqueeze(1).to(device) + 0.99 * next_dist.to(device) * (1-dones.unsqueeze(1).to(device))
    expected_quant = torch.autograd.Variable(expected_quant).to(device)

    quant_idx = torch.sort(dist, 1, descending=False)[1].to(device)

    tau_hat = torch.linspace(0.0, 1.0 - 1. / num_quant, num_quant) + 0.5 / num_quant
    tau_hat = tau_hat.unsqueeze(0).repeat(batch_size, 1)
    quant_idx = quant_idx.cpu().data
    batch_idx = np.arange(batch_size)
    tau = tau_hat[:, quant_idx][batch_idx, batch_idx] # projection matrix

    return tau.to(device), expected_quant.to(device)

# update target network
def update_target_network(current_network : torch.nn.Module, target_network : torch.nn.Module):
    target_network.load_state_dict(current_network.state_dict())