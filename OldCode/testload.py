#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:32:32 2017

@author: nigno
"""

from __future__ import print_function, division

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
        
act_list = ['loc1', 'loc2', 'loc3', 'no_loc']
n_categories = 4

accuracyArray = np.load("accuracy.npy")
confusionArrayNP = np.load("confusionNP.npy")
confusionArrayTensor = np.load("confusionTensor.npy")

meanacc = accuracyArray.mean()

print (meanacc)

confusion = confusionArrayNP.mean(0)

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion)
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + act_list, rotation=90)
ax.set_yticklabels([''] + act_list)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()

print (accuracyArray)
print (confusionArrayNP)
print (confusionArrayTensor)

fig = plt.figure()
plt.plot(accuracyArray, color='red')

plt.show()

raw_input('Press enter to continue: ')