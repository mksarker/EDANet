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
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
###@@@ import others library
from tqdm import tqdm, trange
import argparse
import os
import numpy as np
import shutil
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from confusion_matrix import plot_confusion_matrix
from meter import AverageMeter, ProgressMeter, adjust_learning_rate, accuracy
from loss import CB_loss

### define train function
def train(train_loader, model, device, optimizer, epoch, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # switch to train mode
    model.train()
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader): 
        data_time.update(time.time() - end)       
        # measure data loading time
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        # loss = criterion(output, target)
        samples_per_cls= [279, 5451, 7966]
        loss_type = "focal"
        loss =  CB_loss(target, output, samples_per_cls, 3, loss_type, beta=0.9999, gamma=0.5)
        # measure accuracy and record loss
        acc1, acc3 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))

        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            # progress.print(batch_idx)
            print('Train: Epoch: [{0}][{1}/{2}], Loss {loss.avg:.4f}, Accuracy: {top1.avg:.3f}'.format(epoch, batch_idx, 
                            len(train_loader),loss=losses, top1=top1))


### define validate function
def validate(val_loader, model, device, criterion, args):
    min_val_loss = np.Inf
    n_epochs_stop = 5
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # switch to evaluate mode
    model.eval()
    ## for confusion matrix
    Pr = []
    Tr = []
    correct = 0
    Flag = True

    with torch.no_grad():
        end = time.time()
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # loss = criterion(output, target)
            samples_per_cls= [100, 100, 100]
            loss_type = "focal"
            loss =  CB_loss(target, output, samples_per_cls, 3, loss_type, beta=0.9999, gamma=0.5)
            # measure accuracy and record loss
            acc1, acc3 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))

            ## for confusion matrix
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            ## for confusion matrix
            _, predicted = torch.max(output.data, 1)
            predicted = predicted.cpu().numpy()
            predicted=predicted.reshape((-1,1))
            target = target.cpu().data.numpy()
            # print(target.shape)
            target = target.reshape((-1, 1))
            if Flag==True:
                Pr = predicted
                Tr = target
                Flag=False
            else:
                Pr=np.vstack((Pr,predicted))
                Tr=np.vstack((Tr,target))
            
             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('Test: Loss {loss.avg:.4f}, Accuracy: {top1.avg:.3f}'.format(loss=losses,top1=top1))

    if args.arch.startswith('EDANet'):
        with open(args.root+'/results/'+args.arch+"/Val_Acc.txt", "a") as text_file:
            text_file.write('Test: Loss {loss.avg:.4f}, Accuracy: {top1.avg:.3f}\n'.format(loss=losses,top1=top1))
    return top1.avg, losses.avg

### define final validation for confusion matrix and other results
def final_validate(val_loader, model, device, target_names,num_classes, criterion, args):
    # switch to evaluate mode
    model.eval()
    correct = 0
    Pr = []
    Tr = []
    Flag = True
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # loss= criterion(output, target) 
            samples_per_cls= [100, 100]
            loss_type = "focal"
            loss =  CB_loss(target, output, samples_per_cls, 2, loss_type, beta=0.9999, gamma=0.5)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            ## for calculating confusion matrix
            _, predicted = torch.max(output.data, 1)
            predicted = predicted.cpu().numpy()
            predicted=predicted.reshape((-1,1))
            target = target.cpu().data.numpy()
            # print(target.shape)
            target = target.reshape((-1, 1))
            if Flag==True:
                Pr = predicted
                Tr = target
                Flag=False
            else:
                Pr=np.vstack((Pr,predicted))
                Tr=np.vstack((Tr,target))
    Plot(Pr, Tr, target_names,num_classes, args)


def Plot(target_var, predicted, target_names,num_classes, args):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(target_var, predicted)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    class_names = [target_names[i] for i in range(num_classes)]
    print(class_names)

    ##Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, 
                            title='Confusion matrix, without normalization') #

    if args.arch.startswith('EDANet'):
        plt.savefig(args.root+'/results/'+args.arch+'/Confusion_matrix_WN.pdf',dpi = (300))

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                            title='Normalized confusion matrix') #

    # plt.show()
    if args.arch.startswith('EDANet'):
        plt.savefig(args.root+'/results/'+args.arch+'/Confusion_matrix_Nor.pdf',dpi = (300))

    #### save results
    clf_rep=classification_report(target_var,predicted, target_names=target_names)
    cnf_matrix=confusion_matrix(target_var, predicted)
    if args.arch.startswith('EDANet'):
        file_perf = open(args.root+'/results/'+args.arch+'/performances.txt', 'w')
        file_perf.write("classification Report:\n" + str(clf_rep)
                        + "\n\nConfusion matrix:\n"
                        + str(cnf_matrix)
                        )
        file_perf.close() 

