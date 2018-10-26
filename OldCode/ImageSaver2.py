from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import imageio

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class MirkoDataset(Dataset):
    """Face trajectories dataset."""


        
reader_left = imageio.get_reader('dataRegression/eye0trial1.mp4')
reader_right = imageio.get_reader('dataRegression/eye1trial1.mp4')
metadata = reader_left.get_meta_data()
nframes = metadata['nframes']

print(metadata['fps'])

delta_t = 1. / metadata['fps']

print(nframes)

start_frame = np.loadtxt('dataRegression/start_frame.txt')
ground_truth = np.loadtxt('dataRegression/Ground_truth.csv', delimiter=',')
time_stamp_pos = ground_truth[:, 0] - ground_truth[0, 0]
xyz_pos = ground_truth[:, 1:4]



time_stamp_left_img = np.load('dataRegression/eye0_timestamps.npy')
initial_ts = time_stamp_left_img[int(start_frame)]
time_stamp_left_img = time_stamp_left_img - initial_ts

print(len(time_stamp_left_img))

time_stamp_right_img = np.load('dataRegression/eye1_timestamps.npy')
time_stamp_right_img = time_stamp_right_img - initial_ts

print(len(time_stamp_right_img))

print(len(time_stamp_pos))

count = np.zeros(2, dtype=np.int)
count[0] = int(start_frame)

label_count = 0

dataset_cout = 0

selector = 1

stop_cycle = False
stop_label = False

labels = []

while True:
    ts = np.zeros(2, dtype=np.double)
    ts[0] = time_stamp_left_img[count[0]]
    ts[1] = time_stamp_right_img[count[1]]
    if (ts[selector] - ts[(selector + 1) % 2]) > (delta_t / 2):
        selector = (selector + 1) % 2
    if abs(ts[0] - ts[1]) < (delta_t / 2):
        stop_label = False
        while stop_label == False:
            if label_count >= len(time_stamp_pos) - 1:
                stop_cycle = True
                stop_label = True
            if abs(ts[0] - time_stamp_pos[label_count]) > (delta_t / 2):
                stop_label = True
            label_count +=1
            print('into the while')
        
        if stop_cycle == False:
            
            labels.append(xyz_pos[label_count])
            print(label_count)
            print(xyz_pos[label_count])
            
            image_left = reader_left.get_data(count[0])
            image_right = reader_right.get_data(count[1])
            
            image_name = str(dataset_cout) + '.png'
                
            image_path_left = 'dataRegression/left'
            image_path_right = 'dataRegression/right'
            
            io.imsave(image_path_left + image_name, image_left)
            io.imsave(image_path_right + image_name, image_right)
            
            dataset_cout += 1
        
    count[selector] += 1
    if max(count) >= min((len(time_stamp_left_img), 
                          len(time_stamp_right_img))) - 1 or \
       stop_cycle:
        break
np.save('dataRegression/labels.npy', labels)

#for i in range(40):
#    print(str(time_stamp_left_img[int(start_frame) - i]) + \
#          ' ' + \
#          str(time_stamp_right_img[int(start_frame) - i]))
    
    
#    image_left = reader_left.get_data(start_frame - i)
#    image_right = reader_right.get_data(start_frame -i)
#    
#    plt.figure(1)
#    plt.imshow(image_left)
#    plt.show()
#    
#    plt.figure(2)
#    plt.imshow(image_right)
#    plt.show()
#    
#    raw_input('Press enter to continue: ')
    
