import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2dd(64, 128, 5)
        self.fc = nn.Linear(128 * 5 * 5, 10)
        
    def forward(self, x):
        x = F.ReLU(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.ReLU(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
