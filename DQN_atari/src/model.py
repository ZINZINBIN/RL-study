import numpy as np
import torch
import torch.nn as nn
import random
from gym  import spaces

class DQN(nn.Module):
    def __init__(self, observation_space : spaces.Box, action_space : spaces.Discrete):
        super().__init__()
        assert type(observation_space) == spaces.Box,  "observation_space must be of type Box"
        assert type(action_space) == spaces.Discrete,  "action_space must be of type Discrete"
        assert len(observation_space.shape) == 3, "observation space must have the form channels x  width x height"

        self.conv1 = nn.Conv2d(in_channels=observation_space.shape[0], out_channels=32, kernel_size=8,  stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,  stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,  stride=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
            nn.ReLU(),
            nn.Linear(512, action_space.n)
        )

    def forward(self, inputs): 
        x = nn.functional.relu(self.bn1(self.conv1(inputs)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return  x

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

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        adv = self.fc_advantage(x)
        val = self.fc_value(x)

        # Q-value from dueling DQN
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.n_actions)
        return x