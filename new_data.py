import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
from data import data_transforms, data_transforms_test
import torch
import torchvision
from torchvision import datasets
from model import VitNet
import torch.nn as nn
import numpy as np


parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")


args = parser.parse_args()
use_cuda = torch.cuda.is_available()



# Load models
state_dict1 = torch.load('Vitnet_h_14/model_21.pth')
state_dict2 = torch.load('Vitnet_l_16/model_33.pth')
state_dict3 = torch.load('Vitnet_b_16/model_38.pth')
state_dict4 = torch.load('Vitnet_h_14e2e/model_8.pth')



model1 = VitNet('vith14')
model2 = VitNet('vitl16')
model3 = VitNet('vitb16')
model4 = VitNet('vith14e2e')
#model5 = EfficientNet()

model1.load_state_dict(state_dict1)
model2.load_state_dict(state_dict2)
model3.load_state_dict(state_dict3)
model4.load_state_dict(state_dict4)


def get_conf_labels():

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    #model5.eval()


    if use_cuda:
        print('Using GPU')
        model1.cuda()
        model2.cuda()
        model3.cuda()
        model4.cuda()
        #model5.cuda()
    else:
        print('Using CPU')

    crop = datasets.ImageFolder(args.data + '/Inaturalis_crop/Inaturalist_2021_unlabelled',
                        transform=data_transforms_test )

    test_loader = torch.utils.data.DataLoader(
        crop,
        #torch.utils.data.ConcatDataset([og, crop]),
        shuffle=False, num_workers=1)

    indices=[]
    i = 0
    i_conf=0
    for data, target in test_loader:
        i+=1
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        # Sum output distribuutions
        a = nn.Softmax()
        output1 = a(model1(data))
        output2 = a(model2(data))
        output3 = a(model3(data))
        output4 = a(model4(torchvision.transforms.Resize((518,518))(data)))
        output = a(output1+output2+output3+output4)#+output5

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        if output.data.max(1, keepdim=True).values >0.5:
          i_conf +=1
          indices.append(i)

    print('Number of confident samples: %d' % i_conf)
    
    subset = torch.utils.data.Subset(crop, indices)
    torch.save(subset, 'new_data')



