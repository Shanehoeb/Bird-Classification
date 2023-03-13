import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import torch
from torchvision import datasets
import torch.nn as nn
import torchvision
from model import EfficientNet, VitNet

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
state_dict1 = torch.load('Vitnet_h_14/model_21.pth')
state_dict2 = torch.load('Vitnet_l_16/model_33.pth')
state_dict3 = torch.load('Vitnet_b_16/model_38.pth')
state_dict4 = torch.load('Vitnet_h_14e2e/model_8.pth')



model1 = VitNet('vith14')
model2 = VitNet('vitl16')
model3 = VitNet('vitb16')
model4 = VitNet('vith14e2e')

model1.load_state_dict(state_dict1)
model2.load_state_dict(state_dict2)
model3.load_state_dict(state_dict3)
model4.load_state_dict(state_dict4)

model1.eval()
model2.eval()
model3.eval()
model4.eval()


if use_cuda:
    print('Using GPU')
    model1.cuda()
    model2.cuda()
    model3.cuda()
    model4.cuda()
else:
    print('Using CPU')


from data import data_transforms_test, data_transforms

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
        data = data_transforms_test(pil_loader(test_dir2 + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()

        # Sum output distribuutions
        a = nn.Softmax()
        output1 = a(model1(data))
        output2 = a(model2(data))
        output3 = a(model3(data))
        output4 = a(model4(torchvision.transforms.Resize((518,518))(data)))
        output = output1+output2+output3 + output4

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')

