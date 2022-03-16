'''Policy Optimization Based Algorithm
# list
- A3C
- REINFORCE
- A2C
- PPO
- TPGO
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from torch.distributions import Categorical
from pytorch_model_summary import summary

class ActorCritic(nn.Module):
    def __init__(self, h, w, n_actions, hidden_dims = 128):
        super(ActorCritic, self).__init__()

        h1 = self._conv2d_size_out(h, kernel_size = 8, stride = 4)
        w1 = self._conv2d_size_out(w, kernel_size = 8, stride = 4)

        h2 = self._conv2d_size_out(h1, kernel_size = 4, stride = 2)
        w2 = self._conv2d_size_out(w1, kernel_size = 4, stride = 2)

        h3 = self._conv2d_size_out(h2, kernel_size = 3, stride = 1)
        w3 = self._conv2d_size_out(w2, kernel_size = 3, stride = 1)

        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.n1 = nn.LayerNorm([32,h1,w1])
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.n2 = nn.LayerNorm([64, h2, w2])
        self.conv3 = nn.Conv2d(64,64,kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.n3 = nn.LayerNorm([64, h3, w3])

        convw = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(w, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        convh = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(h, kernel_size = 8, stride = 4), 4, 2), 3, 1)
        linear_input_dim = h3 * w3 * 64

        self.hidden_dims = hidden_dims
        self.n_actions = n_actions

        # policy distirbution : pi(a|s)
        self.actor = nn.Sequential(
            nn.Linear(linear_input_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, n_actions)
        )
        # value : V(s)
        self.critic = nn.Sequential(
            nn.Linear(linear_input_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1)
        )

        # init encoder
        self.init_weights()
        
    def _conv2d_size_out(self, size, kernel_size = 5, stride = 2):
        outputs = (size - (kernel_size - 1) - 1) // stride + 1
        return outputs

    def forward(self, x):
        # Encoding and Decoding
        # x = nn.functional.relu(self.bn1(self.conv1(x)))
        # x = nn.functional.relu(self.bn2(self.conv2(x)))
        # x = nn.functional.relu(self.bn3(self.conv3(x)))

        x = nn.functional.relu(self.n1(self.conv1(x)))
        x = nn.functional.relu(self.n2(self.conv2(x)))
        x = nn.functional.relu(self.n3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        # Value approximation
        value = self.critic(x)

        # Policy Distribution
        policy = nn.functional.softmax(self.actor(x))
        dist = Categorical(policy)

        return value, dist

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))

    def init_weights(self):
        # xavier init
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)

        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)
        nn.init.constant_(self.conv3.bias, 0.0)

        # normalalization
        # nn.init.normal_(self.conv1.weight, mean = 0, std = 0.1)
        # nn.init.normal_(self.conv2.weight, mean = 0, std = 0.1)
        # nn.init.normal_(self.conv3.weight, mean = 0, std = 0.1)

# DPG algorithm


# DDPG algorithm