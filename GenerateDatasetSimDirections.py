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
import cv2
import json
from pprint import pprint

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

#root_dir = '/media/nigno/Data/simulatedEyesDataset/imgs/'
root_dir = '/media/nigno/Data/UnityProjects/UnityEyes/Build/imgs/'
save_root_dir = '/media/nigno/Data/Projects/EyeNet/Datasets/SimDatasetFull/'

dataset_list = os.listdir(root_dir)
num_images = int(len(dataset_list) / 4)

print (num_images)

raw_input('Press enter to continue: ')

class_num = 30.0
class_factor = (2.0 / class_num)

num_elem_per_dir = 1000
dataset_count = 0
labels = []
labels_dir_left = []
labels_dir_right = []
labels_dist = []
labels_class_left = []
labels_class_right = []
for i in range(num_images):
    print(str(i // num_elem_per_dir) + " " + str(dataset_count) )
    
    if (i % num_elem_per_dir) == 0.0:
        dataset_count = 0
        subset_path = str(i // num_elem_per_dir) + '/'
        save_path = save_root_dir + subset_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
#        if ((i // num_elem_per_dir) > 0):
#            subset_path = str((i // num_elem_per_dir) - 1 ) + '/'
#            labels_save_path = save_root_dir + subset_path
#            np.save(labels_save_path + 'labels.npy', labels)
#            np.save(labels_save_path + 'labels_dir_left.npy', labels_dir_left)
#            np.save(labels_save_path + 'labels_dir_right.npy', labels_dir_right)
#            np.save(labels_save_path + 'labels_dist.npy', labels_dist)
        labels = []
        labels_dir_left = []
        labels_dir_right = []
        labels_dist = []
        labels_class_left = []
        labels_class_right = []
    
#    if (i % 2) == 0.0:
#        json_name = str((i * 2) + 1) + '.json'        
#        
#        with open(root_dir + json_name) as json_data:
#            data = json.load(json_data)
#            json_data.close()
#            
#        pitch = float(data['target_3D_pos']['pitch']) * np.pi / 180.0
#        yaw = float(data['target_3D_pos']['yaw']) * np.pi / 180.0
#        d = float(data['target_3D_pos']['z'])
#        x = d * np.cos(pitch) * np.sin(yaw)
#        y = d * np.sin(pitch) * np.cos(yaw)
#        z = d * np.cos(pitch) * np.cos(yaw)
#        labels.append([x, y, z])
    
    
    json_name = str((i * 2) + 1) + '.json'        
    
    with open(root_dir + json_name) as json_data:
        data = json.load(json_data)
        json_data.close()
        
    x = float(data['target_3D_pos']['x'])
    y = float(data['target_3D_pos']['y'])
    z = float(data['target_3D_pos']['z'])
    x *= (z / 20.0)
    y *= (z / 20.0)
    labels.append([x, y, z]) 
    
    dist = float(data['target_3D_pos']['eye_distance'])
    labels_dist.append(dist) 
    
    x = float(data['eye_3D_vec']['x'])
    y = float(data['eye_3D_vec']['y'])
    z = float(data['eye_3D_vec']['z'])
    labels_dir_left.append([x, y, z]) 
    
    x_class = (x + 1.0) // class_factor
    y_class = (y + 1.0) // class_factor
    class_val = int(y_class * class_num + x_class)
    labels_class_left.append(class_val)
    
    json_name = str((i * 2) + 2) + '.json'        
    
    with open(root_dir + json_name) as json_data:
        data = json.load(json_data)
        json_data.close()
        
    x = float(data['eye_3D_vec']['x'])
    y = float(data['eye_3D_vec']['y'])
    z = float(data['eye_3D_vec']['z'])
    labels_dir_right.append([x, y, z]) 
    
    x_class = (x + 1.0) // class_factor
    y_class = (y + 1.0) // class_factor
    class_val = int(y_class * class_num + x_class)
    labels_class_right.append(class_val)
    
    img_name_left = str((i * 2) + 1) + '.jpg'
    img_name_right = str((i * 2) + 2) + '.jpg'
    
    cv_left = cv2.imread(root_dir + img_name_left)
    cv_right = cv2.imread(root_dir + img_name_right)
    
    image_left = cv2.cvtColor(cv_left, cv2.COLOR_BGR2GRAY)
    image_right = cv2.cvtColor(cv_right, cv2.COLOR_BGR2GRAY)
    
    img_name = str(dataset_count) + '.png'  
    
    image_path_left = save_path + 'left'
    image_path_right = save_path + 'right'
    
    io.imsave(image_path_left + img_name, image_left)
    io.imsave(image_path_right + img_name, image_right)
    
    dataset_count += 1
    
    if (((i % num_elem_per_dir) == (num_elem_per_dir - 1)) or (i == (num_images - 1))):
        subset_path = str(i // num_elem_per_dir) + '/'
        labels_save_path = save_root_dir + subset_path
        np.save(labels_save_path + 'labels.npy', labels)
        np.save(labels_save_path + 'labels_dir_left.npy', labels_dir_left)
        np.save(labels_save_path + 'labels_dir_right.npy', labels_dir_right)
        np.save(labels_save_path + 'labels_dist.npy', labels_dist)
        np.save(labels_save_path + 'labels_class_left.npy', labels_class_left)
        np.save(labels_save_path + 'labels_class_right.npy', labels_class_right)

raw_input('Press enter to continue: ')   
        
