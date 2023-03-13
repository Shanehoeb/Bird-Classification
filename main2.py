import argparse
import os
import torch
from torch._C import MobileOptimizerType
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')




args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import data_transforms, data_transforms_test,data_transforms_test_2, data_transforms_2




crop = datasets.ImageFolder(args.data + '/crop_dataset/train_images',
                         transform=data_transforms)
new_d = datasets.ImageFolder('bird_dataset/Extra_pseudo_labeled',
                         transform=data_transforms)                  


                        
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([ new_d, crop]),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

crop_val = datasets.ImageFolder(args.data + '/crop_dataset/val_images_crop',
                         transform=data_transforms_test)


val_loader = torch.utils.data.DataLoader(
    crop_val,
    batch_size=args.batch_size, shuffle=True, num_workers=1)


# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import  VitNet,FVNet2
#model=VitNet('vitl16')
model = FVNet2()
state_dict = torch.load('experiment/model_11.pth')
model.load_state_dict(state_dict)
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.7)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        #print(data)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return 100. * correct / len(val_loader.dataset), validation_loss


best_val=0
it = 0
if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        val,val_loss = validation()
        lr_scheduler.step(val_loss)
        if val>best_val:
          best_val = val
          if val>90:
            model_file = args.experiment + '/model_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), model_file)
            print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
        elif epoch==1:
          model_file = args.experiment + '/model_' + str(epoch) + '.pth'
          torch.save(model.state_dict(), model_file)
          print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')

        else:
          it+=1
          print('it %d without saving'%it)
          if it >20:
            model_file = args.experiment + '/model_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), model_file)
            print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
            it = 0
