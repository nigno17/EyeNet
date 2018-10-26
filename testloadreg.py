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
        

abserrorTrainArray = np.load("abserrorTrain.npy")
abserrorArray = np.load("abserror.npy")

print (abserrorTrainArray)
print (abserrorArray)


fig = plt.figure()
plt.plot(abserrorTrainArray, color='blue')
plt.plot(abserrorArray, color='red')

plt.show()

raw_input('Press enter to continue: ')