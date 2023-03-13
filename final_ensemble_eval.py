import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import torch
from torchvision import datasets
import torch.nn as nn
import torchvision
from model import FVNet, VitNet, FVNet2

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")



args = parser.parse_args()
use_cuda = torch.cuda.is_available()

# Load models

state_dict2 = torch.load('experiment2/model_29_l16_v1.pth')
state_dict4 = torch.load('experiment/model_11_h14.pth')




model2 = FVNet()
model4 = FVNet2()


model2.load_state_dict(state_dict2)
model4.load_state_dict(state_dict4)


model2.eval()

model4.eval()


if use_cuda:
    print('Using GPU')

    model2.cuda()
    model4.cuda()
else:
    print('Using CPU')


from data import data_transforms_test, data_transforms_test_2

test_dir2 = args.data + '/crop_dataset/test_images/mistery_category'


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')



output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir2)):
    if 'jpg' in f:
        data = data_transforms_test_2(pil_loader(test_dir2 + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()

        # Sum output distribuutions
        a = nn.Softmax()
        output2 = a(model2(data))
        output4 = a(model4(torchvision.transforms.Resize((518,518))(data)))
        output = output2 + output4

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')

