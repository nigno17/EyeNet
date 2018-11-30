#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:32:41 2017

@author: nigno
"""

# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

from GazeNet import GazeNetLinear, GazeNetRegDir

from MyLoss import LogLikeLoss

from DataLoading import MirkoDatasetRegNorm, MirkoDatasetRegNormRam, Rescale, ToTensor, Normalize, RandomNoise

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

plt.ion()   # interactive mode

def mse_loss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

train = True
restore = False
# Number of time the network is trained
N_trials = 1

# Data augmentation and normalization for training
# Just normalization for validation
use_gpu = torch.cuda.is_available()

if use_gpu:
    torchType = torch.cuda.FloatTensor
else:
    torchType = torch.FloatTensor

abserrorTrainArray = np.zeros(N_trials)
abserrorArray = np.zeros(N_trials)

time_train_global = 0
time_val_global = 0

for trials in range(N_trials):
    print('-------------' + str(trials) + '-------------')
    
    data_transforms_custom1 = transforms.Compose([Rescale((240, 320)),
                                                  #RandomNoise(var = 0.02),
                                                 #Normalize(mean, std),
                                                  ToTensor()])
    data_transforms_custom2 = transforms.Compose([Rescale((240, 320)),
                                                 #Normalize(mean, std),
                                                  ToTensor()])
    seed = trials + 1
    
    #root_dataset = 'Datasets/simDatasetNetNoRandom/'
    root_dataset = 'Datasets/TestDatasetSimDirections/'
    root_dataset_test = 'Datasets/TestDatasetSimDirections/'
    net_dir = 'Nets/alex_pos_linear/'
    dirNet_dir = 'Nets/vgg_directions/'
    checkpoint_dir = net_dir + 'checkpoints/'
    checkpoint_dirNet_dir = dirNet_dir + 'checkpoints/'
    #root_dataset_test = '/media/nigno/Data/newMirko/'
    #root_dataset_test = 'newDataset/'
    #root_dataset_test = 'DatasetBig/'
    
    # Generating the folders for the training and validation set
    root_dir_list = []
    train_per_lis = []
    for i in range(7):
        root_dir_list.append(root_dataset + str(i) + '/')
        train_per_lis.append(0.7)
    
#    dataset_train = MirkoDatasetReg(root_dir = ['training2/', 'training3/', 'test1/'],
    dataset_train = MirkoDatasetRegNormRam(root_dir = root_dir_list,
                                        transform = data_transforms_custom1,
                                        dset_type='train', seed=seed, 
                                        training_per = train_per_lis,
                                        permuted = True)
    dataset_test = MirkoDatasetRegNormRam(root_dir = root_dir_list,
                                       transform = data_transforms_custom2,
                                       dset_type='val', seed=seed,
                                       training_per = train_per_lis,
                                       permuted = True)
    # NORMALIZATION
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    
    np.save(net_dir + 'mean.npy', mean)
    np.save(net_dir + 'std.npy', std)
    
    print(mean)
    print(std)    
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=200,
                                             shuffle=True, num_workers=6)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2,
                                             shuffle=True, num_workers=6)
    
    print(len(dataset_train))
    print(len(dataset_test))
    
    
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    
    #def imshow(inp, title=None):
    #    """Imshow for Tensor."""
    #    inp = inp.numpy().transpose((1, 2, 0))
    #    inp = np.clip(inp, 0, 1)
    #    plt.imshow(inp)
    #    if title is not None:
    #        plt.title(title)
    #    plt.pause(0.001)  # pause a bit so that plots are updated
    
    def train_model(model, dirNet, criterion, optimizer, scheduler, num_epochs=25, start_epoch=0, loss_list=[], abs_list=[], loss_list_val=[], abs_list_val=[]):
        since = time.time()
    
        best_model_wts = model.state_dict()
        best_abs = 100000
        
        time_train = 0
        time_val = 0
        count_epochs = 0
    
        #for epoch in range(num_epochs):
        while best_abs >= 0.01 and count_epochs < num_epochs:
            epoch = count_epochs           
            
            print('Epoch {}/{}'.format(epoch + start_epoch, num_epochs - 1 + start_epoch))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                since_epoch = time.time()
            #for phase in ['train']:
                if phase == 'train':
                    #scheduler.step()
                    model.train(True)  # Set model to training mode
                    dataloader = dataloader_train
                    dataset_size = len(dataset_train)
                else:
                    model.train(False)  # Set model to evaluate mode
                    dataloader = dataloader_test
                    dataset_size = len(dataset_test)
    
                running_loss = 0.0
                running_abs = 0
    
                # Iterate over data.
                samples_count = 0
                for data in dataloader:
                    samples_count += 1
                    # get the inputs
                    samples = data
    
                    # wrap them in Variable
                    if (use_gpu):
                        img1 = Variable(samples['image_left'].cuda())
                        img2 = Variable(samples['image_right'].cuda())
                        label = Variable(samples['label'].cuda())
                    else:
                        img1 = Variable(samples['image_left'])
                        img2 = Variable(samples['image_right'])
                        label = Variable(samples['label'])
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    left_dir = dirNet(img1)
                    right_dir = dirNet(img2)
                    pred = model(left_dir, right_dir)
                    #loss = criterion(pred, cov, label)
                    loss = criterion(pred, label)
                    
                    # try to change the loss function
                    #gain = Variable(torch.Tensor((1 / label.cpu().data.numpy()[:, 2])).cuda())                                      
                    #loss = torch.sum(gain * torch.sqrt(torch.sum((pred - label).pow(2), 1)))                       
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    # NORMALIZATION
                    if (use_gpu):
                        label = (label * Variable(torch.from_numpy(std).float().cuda())) + Variable(torch.from_numpy(mean).float().cuda())
                        pred = (pred * Variable(torch.from_numpy(std).float().cuda())) + Variable(torch.from_numpy(mean).float().cuda())
                    else:
                        label = (label * Variable(torch.from_numpy(std).float())) + Variable(torch.from_numpy(mean).float())
                        pred = (pred * Variable(torch.from_numpy(std).float())) + Variable(torch.from_numpy(mean).float())
                        
                    abs_loss = torch.sum(torch.sqrt(torch.sum((pred - label).pow(2), 1)))
    
                    # statistics
                    running_loss += label.shape[0] * loss.data[0]
                    #print('sample {}/{}. Loss: {}. Dataset size: {}'.format(samples_count, len(dataloader), loss.data[0], dataset_size))
                    running_abs += float(abs_loss.cpu().data.numpy())
                    printProgressBar(samples_count, len(dataloader), prefix = phase, suffix = 'Complete', length = 50)
                epoch_loss = running_loss / dataset_size
                epoch_abs = running_abs / dataset_size
    
                print('{} Loss: {:.8f} Abs: {:.8f}'.format(phase, epoch_loss, epoch_abs))
                if phase == 'val':
                    loss_list_val += [epoch_loss]
                    abs_list_val += [epoch_abs]
                else:
                    loss_list += [epoch_loss]
                    abs_list += [epoch_abs]
                    
                torch.save({
                        'epoch': epoch + start_epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_abs': best_abs,
                        'loss_list': loss_list,
                        'abs_list': abs_list,
                        'loss_list_val': loss_list_val,
                        'abs_list_val': abs_list_val,
                        'optimizer': optimizer.state_dict(),
                        }, checkpoint_dir + 'checkpointAllEpochs.tar' )
                # deep copy the model
                if ((epoch + start_epoch + 1) % 10) == 0:
                #if phase == 'val' and epoch_abs < best_abs:
                #if phase == 'train' and epoch_acc > best_acc:
                    print('BEST')
                    best_abs = epoch_abs
                    best_model_wts = model.state_dict()
                    torch.save({
                                'epoch': epoch + start_epoch + 1,
                                'state_dict': model.state_dict(),
                                'best_abs': best_abs,
                                'loss_list': loss_list,
                                'abs_list': abs_list,
                                'loss_list_val': loss_list_val,
                                'abs_list_val': abs_list_val,
                                'optimizer': optimizer.state_dict(),
                                }, checkpoint_dir + 'checkpoint' + str(epoch + start_epoch + 1) + '.tar' )
                time_elapsed_epoch = time.time() - since_epoch
                print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed_epoch // 60, time_elapsed_epoch % 60))
                
                if phase == 'train':
                    time_train += time_elapsed_epoch
                else:
                    time_val += time_elapsed_epoch
    
            print()
            count_epochs += 1
    
        time_elapsed = time.time() - since
        print('Training + Val complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_train // 60, time_train % 60))
        print('Val complete in {:.0f}m {:.0f}s'.format(
            time_val // 60, time_val % 60))
        print('Best val Absolute Error: {:4f}'.format(best_abs))
    
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, time_train, time_val
         

    dirNet = GazeNetRegDir(1024)
    if os.path.isfile(checkpoint_dirNet_dir + 'checkpointAllEpochs.tar'):
        print("=> loading checkpoint '{}'".format(checkpoint_dirNet_dir + 'checkpointAllEpochs.tar'))
        checkpoint = torch.load(checkpoint_dirNet_dir + 'checkpointAllEpochs.tar')
        dirNet.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
                  
    dirNet.train(False)
    
    model = GazeNetLinear(6)
    
    if (use_gpu):
        model = model.cuda()
        dirNet = dirNet.cuda()
        
    print(model)
    
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
      
    if (train == True):
#        criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        #criterion = LogLikeLoss
    
        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
        #optimizer_ft = optim.Adam(model.parameters())
    
        # Decay LR by a factor of 0.1 every 7 epochs
        #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        exp_lr_scheduler = None
    
        model, time_train, time_val = train_model(model, dirNet, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=1000)
        time_train_global += time_train
        time_val_global += time_val
    
    if os.path.isfile(checkpoint_dir + 'checkpointAllEpochs.tar'):
        print("=> loading checkpoint '{}'".format(checkpoint_dir + 'checkpointAllEpochs.tar'))
        checkpoint = torch.load(checkpoint_dir + 'checkpointAllEpochs.tar')
        start_epoch = checkpoint['epoch']
        best_abs = checkpoint['best_abs']
        loss_list = checkpoint['loss_list']
        abs_list = checkpoint['abs_list']
        loss_list_val = checkpoint['loss_list_val']
        abs_list_val = checkpoint['abs_list_val']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
            
    if (restore == True):
        #        criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        #criterion = LogLikeLoss
        
        # Observe that all parameters are being optimized
        #optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        #optimizer_ft = optim.Adam(model.parameters())
        
        #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        exp_lr_scheduler = None
        
        model = train_model(model, dirNet, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=1000, start_epoch=start_epoch,
                               loss_list=loss_list, abs_list=abs_list,
                               loss_list_val=loss_list_val, abs_list_val=abs_list_val)

    count = 0
    time_elapsed_single = 0
    
    # Generating the folders for the test set  
    test_root_dir_list = []
    test_per_lis = []
    for i in range(7):
        test_root_dir_list.append(root_dataset_test + str(i) + '/')
        test_per_lis.append(0.0)
    
    dataset_test = MirkoDatasetRegNormRam(root_dir = test_root_dir_list,
                                          transform = data_transforms_custom2,
                                          dset_type='val', seed=seed,
                                          training_per = test_per_lis,
                                          permuted = False)
    
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2,
                                                  shuffle=False, num_workers=8)  
                                                  
    mean = np.load(net_dir + 'mean.npy')
    std = np.load(net_dir + 'std.npy')    
    
    for data in dataloader_test:
        # Go through a bunch of examples and record which are correctly guessed
        start_time = time.time()
        #data = dataset_test[count]
        samples = data
        # wrap them in Variable
        if (use_gpu):
            img1 = Variable(samples['image_left'].cuda())
            img2 = Variable(samples['image_right'].cuda())
            label = Variable(samples['label'].cuda())
        else:
            img1 = Variable(samples['image_left'])
            img2 = Variable(samples['image_right'])
            label = Variable(samples['label'])
        
        start_time = time.time()
        #PROVA-----------------------
        #pred = model(img1.unsqueeze(0), img2.unsqueeze(0))
        pred, cov = model(img1, img2)
        
        if (use_gpu):
            pred_np = pred.cpu().data.numpy()
            cov_np = cov.cpu().data.numpy()
            ground_truth_np = label.cpu().data.numpy()
        else:
            pred_np = pred.data.numpy()
            cov_np = cov.data.numpy()
            ground_truth_np = label.data.numpy()
        
        # NORMALIZATION
        pred_np = (pred_np * std) + mean
        ground_truth_np = (ground_truth_np * std) + mean
            
        if count == 0:
            predictions = pred_np
            ground_truth = ground_truth_np
            covariance = cov_np
        else:
            predictions = np.concatenate((predictions, pred_np), axis=0)
            ground_truth = np.concatenate((ground_truth, ground_truth_np), axis=0)
        
        time_elapsed_single += time.time() - start_time
        count += 1
    
    time_elapsed_single /= count    
    print('Prediction complete in {:.5f}s'.format(time_elapsed_single))
    
    print(predictions)
    
    np.savetxt(net_dir + 'predictions.txt', predictions, delimiter=',') 
    np.savetxt(net_dir + 'ground_truth.txt', ground_truth, delimiter=',') 
    
#    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1,
#                                                   shuffle=False, num_workers=8)
#    
#    
#    dataset_size = len(dataset_test)
#    running_abs = 0
#    data_counter = 0
#    # Go through a bunch of examples and record which are correctly guessed
#    for data in dataloader_train:
##    for i in range(10):
##        data = dataset_test[i]
#        samples = data
#        # wrap them in Variable
#        if data_counter < 500:
#            if (use_gpu):
#                img1 = Variable(samples['image_left'].cuda())
#                img2 = Variable(samples['image_right'].cuda())
#                label = Variable(samples['label'].cuda())
#            else:
#                img1 = Variable(samples['image_left'])
#                img2 = Variable(samples['image_right'])
#                label = Variable(samples['label'])
#            
#            pred = model(img1, img2)
#            running_abs += float(torch.sqrt(torch.sum((pred - label).pow(2))).cpu().data.numpy())
#
#        data_counter += 1
#        
#        
#    
#    epoch_abs = running_abs / dataset_size    
    
    abserrorTrainArray[trials] = abs_list[len(abs_list) - 1]
    
    
#    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
#                                                  shuffle=False, num_workers=8)
#    
#    
#    
#    dataset_size = len(dataset_test)
#    running_abs = 0
#    data_counter = 0
#    # Go through a bunch of examples and record which are correctly guessed
#    for data in dataloader_test:
##    for i in range(10):
##        data = dataset_test[i]
#        samples = data
#        # wrap them in Variable
#        if data_counter < 500:
#            if (use_gpu):
#                img1 = Variable(samples['image_left'].cuda())
#                img2 = Variable(samples['image_right'].cuda())
#                label = Variable(samples['label'].cuda())
#            else:
#                img1 = Variable(samples['image_left'])
#                img2 = Variable(samples['image_right'])
#                label = Variable(samples['label'])
#            
#            pred = model(img1, img2)
#            running_abs += float(torch.sqrt(torch.sum((pred - label).pow(2))).cpu().data.numpy())
#
#        data_counter += 1
#        
#        
#    
#    epoch_abs = running_abs / dataset_size
    
    abserrorArray[trials] = abs_list_val[len(abs_list_val) - 1]
 
time_train_global /= N_trials
time_val_global /= N_trials 

print('Training complete in {:.0f}m {:.0f}s'.format(
    time_train_global // 60, time_train_global % 60))
print('Val complete in {:.0f}m {:.0f}s'.format(
    time_val_global // 60, time_val_global % 60))  

np.save(net_dir + "abserror.npy", abserrorArray)
np.save(net_dir + "abserrorTrain.npy", abserrorTrainArray)

np.savetxt(net_dir + 'abs_list.txt', abs_list, delimiter=',') 
np.savetxt(net_dir + 'abs_list_val.txt', abs_list_val, delimiter=',') 

##visualize_model(model_ft
#
if (train == False):
    fig = plt.figure()
    plt.plot(loss_list, color='red')
    plt.plot(loss_list_val, color='blue')
    fig = plt.figure()
    plt.plot(abs_list, color='red')
    plt.plot(abs_list_val, color='blue')
#plt.plot(loss_list_val, color='blue')
#
raw_input('Press enter to continue: ')
#
##
##visualize_model(model_conv)
##
##plt.ioff()
##plt.show()
##
##plt.pause(2)
##
##
