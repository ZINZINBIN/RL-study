from pytorch_model_summary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features : int, out_features : int, std_init : float = 0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x : torch.Tensor)->torch.Tensor:

        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class PPO(nn.Module):
    def __init__(self, h : int, w : int, n_actions : int, output_dim : int, fc_dims : int = 128, softmax_dim : int = 1):
        super(PPO, self).__init__()
        self.h = h
        self.w = w
        self.output_dim = output_dim
        self.softmax_dim = softmax_dim
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

        self.fc = nn.Sequential(
            nn.Linear(linear_input_dim+1, fc_dims),
            nn.ReLU(),
        )

        self.fc_pi = nn.Sequential(
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions)
        )

        self.fc_v = nn.Sequential(
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, output_dim)
        )

    def _conv2d_size_out(self, size, kernel_size = 5, stride = 2):
        outputs = (size - (kernel_size - 1) - 1) // stride + 1
        return outputs
    
    def forward(self, obs:torch.Tensor, compassAngle:torch.Tensor)->torch.Tensor:
        # obs
        x = nn.functional.relu(self.bn1(self.conv1(obs)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        # concat with compassAngle
        x = torch.cat((x, compassAngle), dim = 1)
        x = self.fc(x)

        prob = self.fc_pi(x)
        log_prob = F.softmax(prob, dim = self.softmax_dim)
        value = self.fc_v(x)

        return log_prob, value

    def summary(self):
        sample_x = torch.zeros((8, 3, self.h, self.w))
        sample_y = torch.zeros((8,1))
        return print(summary(self, sample_x, sample_y, show_input = True, show_hierarchical = False, print_summary = True))