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