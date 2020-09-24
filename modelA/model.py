import torch
import torch.nn as nn
import torch.nn.functional as f

class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        
        # construction
        # input 3, output 64, kernel 3, stride 1(default), padding 0(default)
        self.block1_conv1 = nn.Conv2d(3, 64, 3)
        self.block1_conv2 = nn.Conv2d(64, 64, 3)
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.block2_conv1 = nn.Conv2d(64, 128, 3)
        self.block2_conv2 = nn.Conv2d(128, 128, 3)
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.block3_conv1 = nn.Conv2d(128,256, 3)
        self.block3_conv2 = nn.Conv2d(256,256, 3)
        self.block3_conv3 = nn.Conv2d(256,256, 3)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.block4_conv1 = nn.Conv2d(256,512, 3)
        self.block4_conv2 = nn.Conv2d(512,512, 3)
        self.block4_conv3 = nn.Conv2d(512,512, 3)
        self.pool4 = nn.MaxPool2d(2,2)

        self.block5_conv1 = nn.Conv2d(512,512, 3)
        self.block5_conv2 = nn.Conv2d(512,512, 3)
        self.block5_conv3 = nn.Conv2d(512,512, 3)
        self.pool5 = nn.MaxPool2d(2,2)

        self.avepool = nn.AdaptiveAvgPool2d((7,7))

        self.fc1 = nn.Linear(512*7*7, 4096) 
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,4)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = f.relu(self.block1_conv1(x))
        x = f.relu(self.block1_conv2(x))
        x = self.pool1(x)
        x = f.relu(self.block2_conv1(x))
        x = f.relu(self.block2_conv2(x))
        x = self.pool2(x)
        x = f.relu(self.block3_conv1(x))
        x = f.relu(self.block3_conv2(x))
        x = f.relu(self.block3_conv3(x))
        x = self.pool3(x)
        x = f.relu(self.block4_conv1(x))
        x = f.relu(self.block4_conv2(x))
        x = f.relu(self.block4_conv3(x))
        x = self.pool4(x)
        x = f.relu(self.block5_conv1(x))
        x = f.relu(self.block5_conv2(x))
        x = f.relu(self.block5_conv3(x))
        x = self.pool5(x)

        x = self.avepool(x)

        x = x.view(-1, 512*7*7) # １次元に転置して
        x = self.dropout1(x)
        x = f.relu(self.fc1(x)) # 全結合層へ
        x = f.relu(self.fc2(x))
        x = self.dropout2(x)
        x = f.softmax(self.fc3(x))

        return x
