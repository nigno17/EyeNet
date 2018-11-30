#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Nino Cauli
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

# Data augmentation and normalization for training
# Just normalization for validation
use_gpu = torch.cuda.is_available()

if use_gpu:
    torchType = torch.cuda.FloatTensor
else:
    torchType = torch.FloatTensor


class GazeNet(torch.nn.Module):
    def __init__(self, D_latent = 512, D_actions = 4):
        
        super(GazeNet, self).__init__()
        self.D_features = 256 * 6 * 6
        
        alexNet = models.alexnet(pretrained=False)
        self.features = alexNet.features
        self.latent1 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(D_latent * 2, D_latent),
            nn.ReLU(inplace=True),
            nn.Linear(D_latent, D_actions),
            nn.Softmax(),
        )

    def forward(self, img1, img2):

        features1 = self.features(img1)
        features1 = features1.view(features1.size(0), self.D_features)
        features2 = self.features(img2)
        features2 = features2.view(features2.size(0), self.D_features)
        
        lat1 = self.latent1(features1)
        lat2 = self.latent1(features2)
        
        join_features = torch.cat((lat1, lat2), 1)
        output_pos = self.fc_layers(join_features)
        
        return output_pos
    
class GazeNetMine(torch.nn.Module):
    def __init__(self, D_latent = 512, D_actions = 4):
        
        super(GazeNetMine, self).__init__()
        #self.D_features = 256 * 6 * 6
        self.D_features = 256 * 6 * 9
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.latent1 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(D_latent * 2, D_latent),
            nn.ReLU(inplace=True),
            nn.Linear(D_latent, D_actions),
            nn.Softmax(),
        )

    def forward(self, img1, img2):

        features1 = self.features(img1)
        features1 = features1.view(features1.size(0), self.D_features)
        features2 = self.features(img2)
        features2 = features2.view(features2.size(0), self.D_features)
        
        lat1 = self.latent1(features1)
        lat2 = self.latent1(features2)
        
        join_features = torch.cat((lat1, lat2), 1)
        output_pos = self.fc_layers(join_features)
        
        return output_pos
    
class GazeNetReg(torch.nn.Module):
    def __init__(self, D_latent = 512):
        
        super(GazeNetReg, self).__init__()
        self.D_features = 256 * 6 * 9
        #self.D_features = 256 * 2 * 4
        #self.D_features = 256 * 14 * 19
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.latent1 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(D_latent * 2, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(D_latent, 3),
        )

    def forward(self, img1, img2):

        features1 = self.features(img1)
        features1 = features1.view(features1.size(0), self.D_features)
        features2 = self.features(img2)
        features2 = features2.view(features2.size(0), self.D_features)
        
        lat1 = self.latent1(features1)
        lat2 = self.latent1(features2)
        
        join_features = torch.cat((lat1, lat2), 1)
        output_pos = self.fc_layers(join_features)
        
        return output_pos
        
class GazeNetRegLSTM(torch.nn.Module):
    def __init__(self, D_latent = 512):
        
        super(GazeNetRegLSTM, self).__init__()
        self.D_features = 256 * 6 * 9
        #self.D_features = 256 * 2 * 4
        #self.D_features = 256 * 14 * 19
        
        self.hidden_dim = D_latent * 2
        self.D_latent = D_latent
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.latent1 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(D_latent * 2, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(D_latent, 3),
        )
        
        self.lstm = nn.LSTM(D_latent * 2, self.hidden_dim)
        
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, 1, self.hidden_dim).type(torchType)),
                Variable(torch.zeros(1, 1, self.hidden_dim).type(torchType)))

    def forward(self, img1, img2):

        features1 = self.features(img1)
        features1 = features1.view(features1.size(0), self.D_features)
        features2 = self.features(img2)
        features2 = features2.view(features2.size(0), self.D_features)
        
        lat1 = self.latent1(features1)
        lat2 = self.latent1(features2)
        
        join_features = torch.cat((lat1, lat2), 1)
        
        join_feature_LSTM = join_features.view(join_features.size(0), 1, -1)
        
        lstm_out, self.hidden = self.lstm(join_feature_LSTM, self.hidden)
        
        output_pos = self.fc_layers(lstm_out.view(join_features.size(0), -1))
        
        return output_pos
        
class GazeNetReg2(torch.nn.Module):
    def __init__(self, D_latent = 512):
        
        super(GazeNetReg2, self).__init__()
        #self.D_features = 256 * 6 * 6
        self.D_features = 256 * 6 * 9
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.latent1 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(D_latent * 2, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(D_latent, 3),
        )

    def forward(self, img1, img2):

        features1 = self.features(img1)
        features1 = features1.view(features1.size(0), self.D_features)
        features2 = self.features2(img2)
        features2 = features2.view(features2.size(0), self.D_features)
        
        lat1 = self.latent1(features1)
        lat2 = self.latent1(features2)
        
        join_features = torch.cat((lat1, lat2), 1)
        output_pos = self.fc_layers(join_features)
        
        return output_pos

class GazeNetRegMeanVar(torch.nn.Module):
    def __init__(self, D_latent = 512):
        
        super(GazeNetRegMeanVar, self).__init__()
        #self.D_features = 256 * 6 * 6
        self.D_features = 256 * 6 * 9
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.latent1 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(D_latent * 2, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.mean = nn.Sequential(
            nn.Linear(D_latent, 3)
        )
        self.cov = nn.Sequential(
            nn.Linear(D_latent, 6),
        )
        

    def forward(self, img1, img2):

        features1 = self.features(img1)
        features1 = features1.view(features1.size(0), self.D_features)
        features2 = self.features(img2)
        features2 = features2.view(features2.size(0), self.D_features)
        
        lat1 = self.latent1(features1)
        lat2 = self.latent1(features2)
        
        join_features = torch.cat((lat1, lat2), 1)
        fc_output = self.fc_layers(join_features)
        
        output_mean = self.mean(fc_output)
        output_cov = self.cov(fc_output)
        
        return output_mean, output_cov
        
class GazeNetRegDir(torch.nn.Module):
    def __init__(self, D_latent = 512):
        
        super(GazeNetRegDir, self).__init__()
        #self.D_features = 256 * 6 * 6
        self.D_features = 256 * 6 * 9
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.latent1 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(D_latent, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.mean = nn.Sequential(
            nn.Linear(D_latent, 3)
        )
        

    def forward(self, img):

        features1 = self.features(img)
        features1 = features1.view(features1.size(0), self.D_features)
        
        lat1 = self.latent1(features1)

        fc_output = self.fc_layers(lat1)
        
        output_mean = self.mean(fc_output)
        
        return output_mean
        
class GazeNetLinear(torch.nn.Module):
    def __init__(self, D_latent = 10):
        
        super(GazeNetLinear, self).__init__()
        
        self.fc_layers = nn.Sequential(
            #nn.Dropout(p=0.5),
            nn.Linear(6, D_latent),
            #nn.Dropout(p=0.5),
            nn.Linear(D_latent, 3),
        )
        

    def forward(self, left, right):

        join_features = torch.cat((left, right), 1)

        fc_output = self.fc_layers(join_features)
        
        return fc_output
        
class GazeNetRegVggDir(torch.nn.Module):
    def __init__(self, D_latent = 512):
        
        super(GazeNetRegVggDir, self).__init__()
        #self.D_features = 256 * 6 * 6
        self.D_features = 512 * 7 * 10
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
        )
        self.latent1 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(D_latent, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.mean = nn.Sequential(
            nn.Linear(D_latent, 3)
        )
        

    def forward(self, img):

        features1 = self.features(img)
        features1 = features1.view(features1.size(0), self.D_features)
        
        lat1 = self.latent1(features1)

        fc_output = self.fc_layers(lat1)
        
        output_mean = self.mean(fc_output)
        
        return output_mean
        
class GazeNetRegVggClassDir(torch.nn.Module):
    def __init__(self, D_latent = 512, nClasses = 900):
        
        super(GazeNetRegVggClassDir, self).__init__()
        #self.D_features = 256 * 6 * 6
        self.D_features = 512 * 7 * 10
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
        )
        self.latent1 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(D_latent, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(D_latent, nClasses)
        )
        

    def forward(self, img):

        features1 = self.features(img)
        features1 = features1.view(features1.size(0), self.D_features)
        
        lat1 = self.latent1(features1)

        fc_output = self.fc_layers(lat1)
        
        output_class = self.classifier(fc_output)
        
        return output_class
        
class GazeNetRegVgg(torch.nn.Module):
    def __init__(self, D_latent = 512):
        
        super(GazeNetRegVgg, self).__init__()
        #self.D_features = 256 * 6 * 6
        self.D_features = 512 * 7 * 10
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
        )
        self.latent1 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(D_latent * 2, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(D_latent, 3),
        )

    def forward(self, img1, img2):

        features1 = self.features(img1)
        features1 = features1.view(features1.size(0), self.D_features)
        features2 = self.features(img2)
        features2 = features2.view(features2.size(0), self.D_features)
        
        lat1 = self.latent1(features1)
        lat2 = self.latent1(features2)
        
        join_features = torch.cat((lat1, lat2), 1)
        output_pos = self.fc_layers(join_features)
        
        return output_pos
