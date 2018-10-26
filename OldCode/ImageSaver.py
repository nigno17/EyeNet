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


        
reader_left = imageio.get_reader('data/eye0.mp4')
reader_right = imageio.get_reader('data/eye1.mp4')
metadata = reader_left.get_meta_data()
nframes = metadata['nframes']

print(metadata['fps'])

print(nframes)

for i in range(nframes):

    image_left = reader_left.get_data(i)
    image_right = reader_right.get_data(i)
    
    image_name = str(i) + '.png'
        
    image_path_left = 'data/left'
    image_path_right = 'data/right'

    io.imsave(image_path_left + image_name, image_left)
    io.imsave(image_path_right + image_name, image_right)
    
    
