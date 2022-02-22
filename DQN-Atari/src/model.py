import torch 
import torch.nn as nn
from torch.autograd import Variable

class DQN(nn.Module):
    def __init__(self, h, w, output_dims):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 5, stride = 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,kernel_size = 5, stride = 2)
        self.bn3 = nn.BatchNorm2d(32)

        convw = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(w, kernel_size = 5, stride = 2)))
        convh = self._conv2d_size_out(self._conv2d_size_out(self._conv2d_size_out(h, kernel_size = 5, stride = 2)))
        linear_input_dim = convw * convh * 32

        self.head = nn.Linear(linear_input_dim, output_dims)
        
    def _conv2d_size_out(self, size, kernel_size = 5, stride = 2):
        outputs = (size - (kernel_size - 1) - 1) // stride + 1
        return outputs

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))

        return x

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