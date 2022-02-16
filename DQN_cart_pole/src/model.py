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

'''DDQN : Double Deep Q-Netowrk
(1) BackGround
- over-estimate problem : Q-learning 알고리즘이 특정조건에서 action value가  지나치게 커진다
- Y = R + gamma * max Q(s,a) => R + gamma * Q(s, argmax Q(s,a)) 로 대체한다.
(2) Structure
- with CNN : CNN Q-network를 2개 활용하여 진행
- without CNN : Linear Layer를 활용하여 진행

model은 위 DQN을 그대로 활용
'''



'''Dueling DQN
(1) BackGround
- function estimator : advantage + value로 separate
- Y = R + gamma * max Q(s,a) => R + gamma * Q(s, argmax Q(s,a)) 로 대체한다.
(2) Structure
- Feature Layer
- Value Layer
- Advantage Layer
'''

class DuelingCnnDQN(nn.Module):
    def __init__(self,h,w,output_dims):
        super(DuelingCnnDQN, self).__init__()
        self.h = h
        self.w = w
        self.output_dims = output_dims
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

        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512,output_dims)
        )

    def feature_size(self):
        return self.features(Variable(torch.zeros(1, *self.input_shape))).view(1,-1).size(1)
    

    def _conv2d_size_out(self, size, kernel_size = 5, stride = 2):
        outputs = (size - (kernel_size - 1) - 1) // stride + 1
        return outputs

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))

        return x

