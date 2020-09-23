import torch
import torch.nn as nn
import torch.nn.functional as f

class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        
        # construction
        # input 3, output 6, kernel 5, stride 1(default), padding 0(default)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5) # １次元に転置して
        x = f.relu(self.fc1(x)) # 全結合層へ
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        return x
