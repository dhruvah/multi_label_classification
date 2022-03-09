# from re import X
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from simple_cnn import SimpleCNN, get_fc

class CaffeNetPool5(nn.Module):
    def __init__(self, num_classes=20, inp_size=224, c_dim=3):
        super().__init__()
        self.num_classes = num_classes
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
        
        self.nonlinear = nn.ReLU()
        
        self.pool = nn.MaxPool2d(3, 2)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.flat_dim = (int)(256*(5)*(5))
        
        self.fc1 = nn.Sequential(*get_fc(self.flat_dim, 4096, 'relu'))
        self.fc2 = nn.Sequential(*get_fc(4096,  4096, 'relu'))
        self.fc3 = nn.Sequential(*get_fc(4096, 20, 'none'))
        
    def forward(self, x):
        N = x.size(0)
        x = self.conv1(x)
        x = self.nonlinear(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.nonlinear(x)
        
        x = self.conv4(x)
        x = self.nonlinear(x)
        
        x = self.conv5(x)
        x = self.nonlinear(x)
        x = self.pool(x)
        
        return x


                
