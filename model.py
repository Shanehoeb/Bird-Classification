import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights, regnet_y_32gf, RegNet_Y_32GF_Weights
import torchsummary
from torchvision.models.vision_transformer import vit_h_14, ViT_H_14_Weights, vit_b_16,ViT_B_16_Weights, vit_l_16,ViT_L_16_Weights


nclasses = 20 

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchsummary
from torchvision.models.vision_transformer import vit_h_14, ViT_H_14_Weights, vit_b_16,ViT_B_16_Weights, vit_l_16,ViT_L_16_Weights


nclasses = 20 
class FVNet2(nn.Module):
  def __init__(self):
    super(FVNet2, self).__init__()
    vitnet = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
    for param in vitnet.parameters():
      param.requires_grad = False
    print(vitnet)
    self.vitnet = vitnet
    self.fc1 = nn.Linear(1000, 500)
    self.dropout1 = nn.Dropout(p=0.2)
    self.fc2 = nn.Linear(500, nclasses)
    self.dropout2 = nn.Dropout(p=0.1)

  def forward(self, x):
    x = self.vitnet(x)
    x = self.dropout1(x)
    x = F.relu(self.fc1(x))
    x = self.dropout2(x)
    x = self.fc2(x)
    return x



class FVNet(nn.Module):
  def __init__(self):
    super(FVNet, self).__init__()
    vitnet = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    for param in vitnet.parameters():
      param.requires_grad = False

    self.vitnet = vitnet
    self.fc1 = nn.Linear(1000, 320)
    self.dropout1 = nn.Dropout(p=0.1)
    self.fc2 = nn.Linear(320, nclasses)
    self.dropout2 = nn.Dropout(p=0.1)

  def forward(self, x):
    x = self.vitnet(x)
    x = self.dropout1(x)
    x = F.relu(self.fc1(x))
    x = self.dropout2(x)
    x = self.fc2(x)
    return x

class VitNet(nn.Module):
  def __init__(self, net_name):
    super(VitNet, self).__init__()
    #vitnet = torch.hub.load("facebookresearch/swag", model="vit_h14", pretrained=True, weight_only=True)
    if net_name == 'vitb16':
      vitnet = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
      for param in vitnet.parameters():
        param.requires_grad = False
      vitnet.heads=  nn.Sequential(nn.Linear(in_features=768, out_features=20,bias=True))
    
    if net_name == 'vitl16':
      vitnet = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
      for param in vitnet.parameters():
        param.requires_grad = False
      vitnet.heads=  nn.Sequential(nn.Linear(in_features=1024, out_features=20,bias=True))
    
    if net_name == 'vith14':
      vitnet = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
      for param in vitnet.parameters():
        param.requires_grad = False
      vitnet.heads=  nn.Sequential(nn.Linear(in_features=1280, out_features=20,bias=True))

    if net_name == 'vith14e2e':
      vitnet = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
      for param in vitnet.parameters():
        param.requires_grad = False
      vitnet.heads=  nn.Sequential(nn.Linear(in_features=1280, out_features=20,bias=True))
      
    if net_name == 'vitl16e2e':
      vitnet = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
      for param in vitnet.parameters():
        param.requires_grad = False
      vitnet.heads=  nn.Sequential(nn.Linear(in_features=1024, out_features=20,bias=True))


    self.vitnet = vitnet
    self.dropout1 = nn.Dropout(p=0.5)
    self.fc1 = nn.Linear(20, nclasses)

  def forward(self, x):
    x = self.vitnet(x)
    x = self.dropout1(x)
    x = F.relu(self.fc1(x))
    

    return x
