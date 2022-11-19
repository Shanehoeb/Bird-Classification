import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.linear = nn.Linear(resnet.fc.in_features, nclasses)
        self.dropout1 = nn.Dropout(p=0.9)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = F.relu(self.linear(x))
        return x

