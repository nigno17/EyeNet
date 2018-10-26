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


root_dir = '/media/nigno/Data/dataComplete/013/'
save_root_dir = '/home/nigno/Robots/pytorch_tests/MirkoNet/three_markers3/'

        
reader_left = imageio.get_reader(root_dir + 'eye0.mp4')
reader_right = imageio.get_reader(root_dir + 'eye1.mp4')
metadata = reader_left.get_meta_data()
nframes = metadata['nframes']

print(metadata['fps'])

delta_t = 1. / metadata['fps']

print(nframes)


time_stamp_left_img = np.load(root_dir + 'eye0_timestamps.npy')
time_stamp_left_img = time_stamp_left_img

print(len(time_stamp_left_img))

time_stamp_right_img = np.load(root_dir + 'eye1_timestamps.npy')
time_stamp_right_img = time_stamp_right_img


print(len(time_stamp_right_img))

num_images = min(len(time_stamp_right_img), len(time_stamp_left_img)) 

print (num_images)

raw_input('Press enter to continue: ')


dataset_count = 0
for_count = 0
labels = []
for i in range(num_images):
    #if (for_count % 10) == 0.0:
    if True:
        print(dataset_count)
        labels.append([0, 0, 0])    
        image_left = reader_left.get_data(i)
        image_right = reader_right.get_data(i)
        
        image_name = str(dataset_count) + '.png'
            
        image_path_left = save_root_dir + 'left'
        image_path_right = save_root_dir + 'right'
        
        io.imsave(image_path_left + image_name, image_left)
        io.imsave(image_path_right + image_name, image_right)
        
        dataset_count += 1
    for_count +=1
        
np.save(save_root_dir + 'labels.npy', labels)  

raw_input('Press enter to continue: ')   
        
