### '' @ This code is Developed by: 
"""
@author: Md Mostafa Kamal Sarker
@ email: m.kamal.sarker@gmail.com
@ Date: 17.05.2020
"""

###@@@ import pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchnet.meter as meter
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
###@@@ import sklearn library
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix
###@@@ import others library
from tqdm import tqdm, trange
import argparse
import os
import shutil
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from confusion_matrix import plot_confusion_matrix
from loss import CB_loss
####@@@ import models here
from edanet import EDANet
## trainer modules
from trainer import train, validate, final_validate
#### for ploting confusin matrix 
from trainer import Plot

###@@@ set the parser arguments
parser = argparse.ArgumentParser(description='EDANet PyTorch Model Training')
### set the learning rate
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--root', default='/COVID-19/EDANet', type=str,  
                    help='Path of the code')   
parser.add_argument('--data_dir', default='/COVID-19/COVIDx-data/covidXdata', type=str,  
                    help='Path of the dataset')  
parser.add_argument('--arch', metavar='ARCH', default='EDANet',
                    help='model architecture (default: EDANet)')

args = parser.parse_args()

###@@@@@@ initialization
torch.manual_seed(args.seed)
## cuda
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

### dataset initilizations
### create label list from diractory of train or test
target_names=os.listdir(os.path.join(args.data_dir, 'train'))
target_names_val=os.listdir(os.path.join(args.data_dir, 'val'))
print (target_names)
print (target_names_val)
num_classes=len(target_names)

### define main function 
def main():
    global args
    best_acc = 0
    # Model
    print('==> Building model..')
    
    if args.arch.startswith('EDANet'):
        net = EDANet(num_classes=num_classes).to(device)
    else:
        print('Please define the model')

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    ### set optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = CB_loss().to(device)
    ### Data loading for train and val 
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    ###
    
    ## for color image(r,g,b)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    ## for gray image
    normalize  = transforms.Normalize([0.5], [0.5]) 
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True)
    # print (train_loader)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True)
    
    # print (val_loader)
    if args.evaluate:
        validate(val_loader, net, device, criterion,args)
        return
    ### 
    # scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True )
    for epoch in range(args.start_epoch, args.epochs):
        ### train for one epoch
        train(train_loader, net,device, optimizer, epoch, criterion, args )
        ### evaluate on validation set
        acc, loss = validate(val_loader, net,device, criterion, args)
        ### remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'state_dict': net.state_dict(),
        }, is_best)
        scheduler.step(loss)
        ## final
    ### final validation and save confusion matrix and other results
    final_validate(val_loader, net, device, target_names,num_classes, criterion, args)

### for saving results in model diroctories
if args.arch.startswith('EDANet'): 
    file_path= args.root+'/results/'+args.arch 
    if not os.path.exists(file_path):
        os.mkdir(file_path) 

### For saving save_checkpoint of model 
def save_checkpoint(state, is_best, filename= file_path+'/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, file_path+'/best_checkpoint.pth.tar')

####### run main
if __name__ == '__main__':
    main()