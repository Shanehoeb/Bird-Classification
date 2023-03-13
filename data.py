import zipfile
import os

import numpy as np
import torchvision.transforms as transforms
import PIL.Image as Image
from augment import fb_rcnn
# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set



data_transforms_2 = transforms.Compose([
    #transforms.Lambda(fb_rcnn),
    #transforms.Resize((224,224)),
    transforms.Resize((512,512)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    #transforms.RandomEqualize(p=0.5),
    transforms.ToTensor(),  # to range [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

data_transforms = transforms.Compose([
    #transforms.Lambda(fb_rcnn),
    #transforms.Resize((224,224)),
    transforms.Resize((518,518)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    #transforms.RandomEqualize(p=0.5),
    transforms.ToTensor(),  # to range [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

data_transforms_test = transforms.Compose([
    #transforms.Lambda(fb_rcnn),
    #transforms.Resize((224,224)),
    transforms.Resize((518,518)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(20),
    transforms.ToTensor(),  # to range [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


data_transforms_test_2 = transforms.Compose([
    #transforms.Lambda(fb_rcnn),
    #transforms.Resize((224,224)),
    transforms.Resize((512,512)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(20),
    transforms.ToTensor(),  # to range [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

