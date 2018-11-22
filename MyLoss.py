# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import torch.nn as nn

def LogLikeLoss(output_mean, output_cov, target):
    D = 3
    torchType = torch.cuda.FloatTensor
    loss = 0
    
    mean = output_mean
    L_vect = output_cov
    numEl = L_vect.size()[0]
    
    #L = Variable(torch.zeros(L_vect.shape()[0], D, D).type(torchType))
    #cov = Variable(torch.ones(L_vect.shape()[0], D, D).type(torchType))
    for i in range(numEl):
        L = Variable(torch.zeros(D, D).type(torchType))
        cov = Variable(torch.ones(D, D).type(torchType))
        #L[i, np.tril_indices(D, 1)] = L_vect.clone()
        L[0, 0] = torch.exp(L_vect[i, 0]).clone()
        L[1, 0] = L_vect[i, 1].clone()
        L[1, 1] = torch.exp(L_vect[i, 2]).clone()
        L[2, 0] = L_vect[i, 3].clone()
        L[2, 1] = L_vect[i, 4].clone()
        L[2, 2] = torch.exp(L_vect[i, 5]).clone()
        cov = L.mm(L.t())
        m = MultivariateNormal(mean[i], cov)
        loss -= m.log_prob(target[i])
    loss /= numEl
        
    
    
#    L = Variable(torch.zeros(D, D).type(torchType))
#    cov = Variable(torch.ones(D, D).type(torchType))
#    for i in range(ngauss):
#        L[i, 0, 0] = torch.exp(cov2[(i * 6) + 0]).clone()
#        L[i, 1, 0] = cov2[(i * 6) + 1].clone()
#        L[i, 1, 1] = torch.exp(cov2[(i * 6) + 2]).clone()
#        L[i, 2, 0] = cov2[(i * 6) + 3].clone()
#        L[i, 2, 1] = cov2[(i * 6) + 4].clone()
#        L[i, 2, 2] = torch.exp(cov2[(i * 6) + 5]).clone()
#        Ltemp = L[i].clone()
#        cov[i] = Ltemp.mm(Ltemp.t())
    
    
    return loss
