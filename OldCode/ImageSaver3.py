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


root_dir = '/media/nigno/Data/mirko dataset/008/'
save_root_dir = 'newDataset/test_mirko_hm/'

        
reader_left = imageio.get_reader(root_dir + 'eye0.mp4')
reader_right = imageio.get_reader(root_dir + 'eye1.mp4')
metadata = reader_left.get_meta_data()
nframes = metadata['nframes']

print(metadata['fps'])

delta_t = 1. / metadata['fps']

print(nframes)


ground_truth = np.loadtxt(root_dir + 'test_ot_nino2.csv', delimiter=',')
ground_truth_rel = np.loadtxt(root_dir + 'test_ot_nino_rel.csv', delimiter=',')
time_stamp_pos = ground_truth[:, 0]
xyz_pos_world = ground_truth[:, 1:4]
xyz_pos = ground_truth_rel[:, 1:4]
sync_eye_ts = ground_truth[:, 11:]


time_stamp_left_img = np.load(root_dir + 'eye0_timestamps.npy')
time_stamp_left_img = time_stamp_left_img

print(len(time_stamp_left_img))

time_stamp_right_img = np.load(root_dir + 'eye1_timestamps.npy')
time_stamp_right_img = time_stamp_right_img


print(len(time_stamp_right_img))

print(len(time_stamp_pos))
print(sync_eye_ts[494, 0])
print(time_stamp_left_img[0])
raw_input('Press enter to continue: ')

count = np.zeros(2, dtype=np.int)

label_count = 1
for i in range(len(sync_eye_ts)):
    if (abs(sync_eye_ts[i, 0] - time_stamp_left_img[0]) < 0.0001 or \
       abs(sync_eye_ts[i, 1] - time_stamp_right_img[0]) < 0.0001) and \
       sync_eye_ts[i, 0] >= 0 and sync_eye_ts[i, 1] >= 0:
          label_count = i
          i = len(sync_eye_ts)
print(label_count)
print(len(sync_eye_ts))

print(min((len(time_stamp_left_img), len(time_stamp_right_img))))

raw_input('Press enter to continue: ')

dataset_count = 0

stop_cycle = False

labels = []

while stop_cycle == False:
    if sync_eye_ts[label_count, 0] >= 0 and sync_eye_ts[label_count, 1] >= 0 and \
       (xyz_pos_world[label_count, 0] != 0 or xyz_pos_world[label_count, 1] != 0 or xyz_pos_world[label_count, 2] != 0):
        while abs(sync_eye_ts[label_count, 0] - time_stamp_left_img[count[0]]) > 0.0001 and \
              count[0] < len(time_stamp_left_img) - 1:
            count[0] += 1
        while abs(sync_eye_ts[label_count, 1] - time_stamp_right_img[count[1]]) > 0.0001 and \
              count[1] < len(time_stamp_right_img) - 1:
            print(abs(sync_eye_ts[label_count, 1] - time_stamp_right_img[count[1]]))
            count[1] += 1
            
        if max(count) < min((len(time_stamp_left_img), 
                             len(time_stamp_right_img))) - 1:
            labels.append(xyz_pos[label_count])
            print(label_count)
            print(xyz_pos[label_count])
            
            image_left = reader_left.get_data(count[0])
            image_right = reader_right.get_data(count[1])
            
            image_name = str(dataset_count) + '.png'
                
            image_path_left = save_root_dir + 'left'
            image_path_right = save_root_dir + 'right'
            
            io.imsave(image_path_left + image_name, image_left)
            io.imsave(image_path_right + image_name, image_right)
            
            dataset_count += 1
        
    label_count += 1
        
    if max(count) >= min((len(time_stamp_left_img), 
                          len(time_stamp_right_img))) - 1 or \
       label_count >= len(sync_eye_ts):
        stop_cycle = True
        
np.save(save_root_dir + 'labels.npy', labels)     
        

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
    
