from turtle import forward
import torch 
import torch.nn as nn
import random
import numpy as np
from torch.autograd import Variable
from torch.distributions import Categorical
from pytorch_model_summary import summary
from typing import Union, Tuple, List

# Deep Q-Network 
class DQN(nn.Module):
    def __init__(self, h, w, output_dims, hidden_dims = 128):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(64)

        convw = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(w, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        convh = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(h, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        linear_input_dim = convw * convh * 64

        self.hidden_dims = hidden_dims

        self.head = nn.Sequential(
            nn.Linear(linear_input_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )
        
    def _conv2d_size_out(self, size, kernel_size = 5, stride = 2):
        outputs = (size - (kernel_size - 1) - 1) // stride + 1
        return outputs

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))

        return x

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))

# Dueling-DQN 
class DuelingDQN(nn.Module):
    def __init__(self, h : int, w : int, n_actions : int, output_dims : int, fc_dims : int = 128):
        super(DuelingDQN, self).__init__()
        self.h = h
        self.w = w
        self.output_dims = output_dims
        self.fc_dims = fc_dims
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 5, stride = 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,kernel_size = 5, stride = 2)
        self.bn3 = nn.BatchNorm2d(32)

        convw = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(w, kernel_size = 5, stride = 2)))
        convh = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(h, kernel_size = 5, stride = 2)))
        linear_input_dim = convw * convh * 32

        self.fc_value = nn.Sequential(
            nn.Linear(linear_input_dim, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, output_dims)
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(linear_input_dim, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, output_dims)
        )

    def _conv2d_size_out(self, size, kernel_size = 5, stride = 2):
        outputs = (size - (kernel_size - 1) - 1) // stride + 1
        return outputs

    def forward(self, x : torch.Tensor)->torch.Tensor:
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        adv = self.fc_advantage(x)
        val = self.fc_value(x)

        # Q-value from dueling DQN
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.n_actions)
        return x


# Noisy Networks : https://arxiv.org/abs/1706.10295
# component : NoisyLinear, NoisyDQN
import math
class NoisyLinear(nn.Module):
    def __init__(self, in_features : int, out_features : int, std_init :float = 0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer(
            'weight_epsilon', torch.FloatTensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer(
            'bias_epsilon', torch.FloatTensor(out_features)
        )

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))

        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return nn.functional.linear(x, weight, bias) # XW + B

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range) # torch.nn.Parameter의 메서드, 1 / 2 *mu_range만큼의 균일 확률 분포
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))


    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # torch.ger() => torch.outer(), outer product of vec1 and vec2
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        # x.sign() -> 원소가 양수면 1, 음수면 -1, 0이면 0
        # Norm 분포로 추출한 원소로 구성된 x => 1,0,-1로 구성된 벡터 생성
        # x의 절대값의 제곱근 형태의 벡터 생성
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt()) 
        return x

class NoiseDQN(nn.Module):
    def __init__(self, h, w, output_dims, hidden_dims = 128):
        super(NoiseDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(64)

        convw = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(w, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        convh = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(h, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        linear_input_dim = convw * convh * 64

        self.hidden_dims = hidden_dims

        self.noisy1 = NoisyLinear(linear_input_dim, hidden_dims)
        self.noisy2 = NoisyLinear(hidden_dims, hidden_dims)
        self.head = nn.Linear(hidden_dims, output_dims)

    def _conv2d_size_out(self, size, kernel_size = 5, stride = 2):
        outputs = (size - (kernel_size - 1) - 1) // stride + 1
        return outputs

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))

        x = nn.functional.relu(self.noisy1(x.view(x.size(0), -1)))
        x = nn.functional.relu(self.noisy2(x))
        x = self.head(x)
        return x

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))

    def act(self, state):
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile = True) # forward만 진행 : volatile = true
        q = self.forward(state)
        action = q.max(1)[1].data[0]
        return action
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

# Categorical DQN : A Distributional Perspective on Reinforcement Learning
# using NoiseLinear Layer for exploration
# archiv : https://arxiv.org/pdf/1707.06887.pdf
class CategoricalDQN(nn.Module):
    def __init__(self, h : int, w : int, output_dims : int, num_atoms : int, V_min :int, V_max : int, hidden_dims = 128):
        super(CategoricalDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(64)

        convw = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(w, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        convh = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(h, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        linear_input_dim = convw * convh * 64

        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.num_atoms = num_atoms
        self.V_min = V_min
        self.V_max = V_max

        self.noisy1 = NoisyLinear(linear_input_dim, hidden_dims)
        self.noisy2 = NoisyLinear(hidden_dims, output_dims * num_atoms)
       
    def _conv2d_size_out(self, size, kernel_size = 5, stride = 2):
        outputs = (size - (kernel_size - 1) - 1) // stride + 1
        return outputs

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))

        x = nn.functional.relu(self.noisy1(x.view(x.size(0), -1)))
        x = self.noisy2(x)
        x = nn.functional.softmax(x.view(-1, self.num_atoms), 1).view(-1, self.output_dims, self.num_atoms)
        return x

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))

    def act(self, state:torch.Tensor):
        with torch.no_grad():
            # state = Variable(state, volatile = True) # forward만 진행 : volatile = true
            dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.V_min, self.V_max, self.num_atoms)
        # action = dist.sum(2).max(1)[1].numpy()[0]
        action = dist.sum(2).max(1)[1].cpu()

        return action

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

# RainbowDQN : DDQN + Dueling DQN + PER + Distributed DQN
# 다양한 DQN 알고리즘을 결합하여 성능을 끌어올린 모델
class RainbowDQN(nn.Module):
    def __init__(self, h : int, w : int, output_dims : int, num_atoms : int, V_min :int, V_max : int, hidden_dims = 128):
        super(RainbowDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(64)

        convw = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(w, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        convh = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(h, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        linear_input_dim = convw * convh * 64

        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.num_atoms = num_atoms
        self.V_min = V_min
        self.V_max = V_max

        # value approximator
        self.noisy1_v = NoisyLinear(linear_input_dim, hidden_dims)
        self.noisy2_v = NoisyLinear(hidden_dims, num_atoms)

        # advantage approximator
        self.noisy1_ad = NoisyLinear(linear_input_dim, hidden_dims)
        self.noisy2_ad = NoisyLinear(hidden_dims, output_dims * num_atoms)
       
    def _conv2d_size_out(self, size, kernel_size = 5, stride = 2):
        outputs = (size - (kernel_size - 1) - 1) // stride + 1
        return outputs

    def forward(self, x:torch.Tensor):

        batch_size = x.size(0)

        # process 1 : inputs -> embedding vector
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))

        # process 2 : separate value and advantage approximator
        value = nn.functional.relu(self.noisy1_v(x.view(x.size(0), -1)))
        value = self.noisy2_v(value)

        advantage = nn.functional.relu(self.noisy1_ad(x.view(x.size(0), -1)))
        advantage = self.noisy2_ad(advantage)

        value = value.view(batch_size, 1, self.output_dims)
        advantage = advantage.view(batch_size, self.output_dims, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = nn.functional.softmax(x.view(-1, self.num_atoms), 1).view(-1, self.num_actions, self.num_atoms)

        return x

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))

    def act(self, state:torch.Tensor):
        with torch.no_grad():
            dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.V_min, self.V_max, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]

        return action

    def reset_noise(self):
        self.noisy1_v.reset_noise()
        self.noisy2_v.reset_noise()
        self.noisy1_ad.reset_noise()
        self.noisy2_ad.reset_noise()
   

# QR-DQN : Distributed Reinforcement Learning with Quantile Regression
# arxiv : https://arxiv.org/pdf/1710.10044.pdf
class QR_DQN(nn.Module):
    def __init__(self, h : int, w : int, output_dims : int, num_quants : int, hidden_dims = 128):
        super(QR_DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(64)

        convw = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(w, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        convh = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(h, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        linear_input_dim = convw * convh * 64

        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        self.num_quants = num_quants

        self.head = nn.Sequential(
                nn.Linear(linear_input_dim, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, output_dims * self.num_quants)
        )
       
    def _conv2d_size_out(self, size, kernel_size = 5, stride = 2):
        outputs = (size - (kernel_size - 1) - 1) // stride + 1
        return outputs

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.head(x)
        x = x.view(x.size(0), self.output_dims, self.num_quants)
        return x

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))

    def get_q_value(self, x):
        x = self.forward(x)
        return x.mean(2) # (batch size, num actions, num quants) => (batch_size, num_actions)

    def act(self, state:Union[torch.Tensor, np.array], epsilon : float = 0.01)->torch.Tensor:

        if len(state.size()) <= 3:
            state = state.unsqueeze_(0)

        if type(state) == np.array:
            state = torch.from_numpy(state)

        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.forward(state).mean(2)
                action = q_values.max(1)[1].cpu()
                #action = action.data.cpu().numpy()[0]
              
        else:
            action = random.randrange(self.output_dims)
            action = torch.tensor(action, dtype = torch.int32, device = 'cpu')
           

        return action

#Hierarchical DQN
class HDQN(nn.Module):
    def __init__(self, h : int, w : int, output_dims : int, hidden_dims = 128):
        super(HDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(64)

        convw = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(w, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        convh = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(h, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        linear_input_dim = convw * convh * 64

        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        self.head = nn.Sequential(
                nn.Linear(linear_input_dim, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, output_dims)
        )
       
    def _conv2d_size_out(self, size, kernel_size = 5, stride = 2):
        outputs = (size - (kernel_size - 1) - 1) // stride + 1
        return outputs

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.head(x)
        x = x.view(x.size(0), self.output_dims, self.num_quants)
        return x

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))

    def get_q_value(self, x):
        x = self.forward(x)
        return x.mean(2) # (batch size, num actions, num quants) => (batch_size, num_actions)

    def act(self, state:Union[torch.Tensor, np.array], epsilon : float = 0.01)->torch.Tensor:

        if len(state.size()) <= 3:
            state = state.unsqueeze_(0)

        if type(state) == np.array:
            state = torch.from_numpy(state)

        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.forward(state).mean(2)
                action = q_values.max(1)[1].cpu()
                #action = action.data.cpu().numpy()[0]
              
        else:
            action = random.randrange(self.output_dims)
            action = torch.tensor(action, dtype = torch.int32, device = 'cpu')
           

        return action

# Encoder for state with 2D image
class Encoder(nn.Module):
    def __init__(self, h : int, w : int):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(64)

        convw = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(w, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        convh = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(h, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        linear_input_dim = convw * convh * 64
        self.linear_input_dim = linear_input_dim

    def _conv2d_size_out(self, size : int, kernel_size : int = 5, stride : int = 2)->int:
        outputs = (size - (kernel_size - 1) - 1) // stride + 1
        return outputs

    def forward(self, x : torch.Tensor)->torch.Tensor:
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

# General Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, h : int, w : int, hidden : int, n_actions : int):
        super(PolicyNetwork, self).__init__()
        self.encoder = Encoder(h,w)
        linear_input_dim = self.encoder.linear_input_dim
        self.linear_input_dim = linear_input_dim
        self.mlp = nn.Sequential(
            nn.Linear(linear_input_dim, hidden),
            # nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            # nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, n_actions)
        )

    def forward(self, x : torch.Tensor)->torch.Tensor:
        x = x.float()
        x = self.encoder(x)
        x = self.mlp(x)
        policy = nn.functional.softmax(x, dim  = 1)
        dist = Categorical(policy)
        return dist

    def select_action(self, state : torch.Tensor)->torch.Tensor:
        x = state.float()
        with torch.no_grad():
            x = self.encoder(x)
            x = self.mlp(x)
            policy = nn.functional.softmax(x, dim  = 1)
            dist = Categorical(policy)
            action = dist.sample()
        return action