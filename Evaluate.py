#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:32:41 2017

@author: nigno
"""

# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

from GazeNet import GazeNetRegMeanVar

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

# Data augmentation and normalization for training
# Just normalization for validation
use_gpu = torch.cuda.is_available()

if use_gpu:
    torchType = torch.cuda.FloatTensor
else:
    torchType = torch.FloatTensor
    
data_transforms_custom1 = transforms.Compose([Rescale((240, 320)),
                                              #RandomNoise(var = 0.02),
                                             #Normalize(mean, std),
                                              ToTensor()])
data_transforms_custom2 = transforms.Compose([Rescale((240, 320)),
                                             #Normalize(mean, std),
                                              ToTensor()])
seed = 1

root_dataset_test = 'Datasets/TestDatasetSim/'
net_dir = 'Nets/LoglikeNoRandomLr3Wd5/'
checkpoint_dir = net_dir + 'checkpoints/'
     

model = GazeNetRegMeanVar(64)
if (use_gpu):
    model = model.cuda()
    
print(model)

optimizer_ft = optim.Adam(model.parameters())

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

count = 0
time_elapsed_single = 0

# Generating the folders for the test set  
test_root_dir_list = []
test_per_lis = []
for i in range(45):
    test_root_dir_list.append(root_dataset_test + str(i) + '/')
    test_per_lis.append(0.0)

dataset_test = MirkoDatasetRegNorm(root_dir = test_root_dir_list,
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


np.savetxt(net_dir + 'abs_list.txt', abs_list, delimiter=',') 
np.savetxt(net_dir + 'abs_list_val.txt', abs_list_val, delimiter=',') 


fig = plt.figure()
plt.plot(loss_list, color='red')
plt.plot(loss_list_val, color='blue')
fig = plt.figure()
plt.plot(abs_list, color='red')
plt.plot(abs_list_val, color='blue')

raw_input('Press enter to continue: ')
