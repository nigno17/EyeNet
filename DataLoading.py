from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, util
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import imageio
import cv2

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

def show_trajectories(image, trajectories):
    plt.scatter(trajectories[:, 0], trajectories[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image_left, image_right, label = sample['image_left'], \
                                         sample['image_right'], \
                                         sample['label']

        new_h, new_w = self.output_size

        img_l = transform.resize(image_left, (new_h, new_w))
        img_r = transform.resize(image_right, (new_h, new_w))

        return {'image_left': img_l, 
                'image_right': img_r, 
                'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image_left, image_right, label = sample['image_left'], \
                                         sample['image_right'], \
                                         sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_left = image_left.transpose((2, 0, 1))
        image_right = image_right.transpose((2, 0, 1))
        return {'image_left': torch.from_numpy(image_left).float(),
                'image_right': torch.from_numpy(image_right).float(),
                'label': torch.from_numpy(label).float()}
        
class Normalize(object):
    """Convert ndarrays in sample to Tensors.
    
    Args:
        mean: Mean of rgb channels
        std: Standard deviation of rgb channels
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        

    def __call__(self, sample):
        image_left, image_right, label = sample['image_left'], \
                                         sample['image_right'], \
                                         sample['label']

        image_left = (image_left - self.mean) / self.std
        image_right = (image_right - self.mean) / self.std
        
        #trajectories = (trajectories - trajectories.mean()) / trajectories.std()
        #print('mean: ' + str(trajectories.mean()))
        #print('mean: ' + str(trajectories.std()))
        
        return {'image_left': image_left, 
                'image_right': image_right, 
                'label': label}
        
class RandomNoise(object):
    """Convert ndarrays in sample to Tensors.
    
    Args:
        mean: Mean of rgb channels
        std: Standard deviation of rgb channels
    """
    
    def __init__(self, mean = 0.0, var = 0.01):
        self.mean = mean
        self.var = var
        

    def __call__(self, sample):
        image_left, image_right, label = sample['image_left'], \
                                         sample['image_right'], \
                                         sample['label']

        img_l = util.random_noise(image_left, mean = self.mean, var = self.var)
        img_r = util.random_noise(image_right, mean = self.mean, var = self.var)
        
        #trajectories = (trajectories - trajectories.mean()) / trajectories.std()
        #print('mean: ' + str(trajectories.mean()))
        #print('mean: ' + str(trajectories.std()))
        
        return {'image_left': img_l, 
                'image_right': img_r, 
                'label': label}
    
class MirkoDataset(Dataset):
    """Face trajectories dataset."""

    def __init__(self, root_dir, transform=None, dset_type='train', seed=1):
        """
        Args:
            indices_file (string): Path to the txt file with image indices.
            traj_dir (string): Directory with all the trajectories.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        nframes = int((len(os.listdir(root_dir)) - 2) / 2)

        np.random.seed(seed)
        permuted_indeces = np.random.permutation(range(nframes))
        
        train_number = int(nframes * 0.7)
        if dset_type == 'train':
            self.indices = permuted_indeces[:train_number]
        else:
            self.indices = permuted_indeces[train_number:]
            
        

        self.fps = 120
        self.root_dir = root_dir
        self.transform = transform
        self.n_lables = 4

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_name = str(self.indices[idx]) + '.png'
        image_left = io.imread(self.root_dir + 'left' + img_name)
        image_right = io.imread(self.root_dir + 'right' + img_name)
        
        if self.indices[idx] < 504:
            label_int = 0
        elif self.indices[idx] < 1032:
            label_int = 1
        elif self.indices[idx] < 1572:
            label_int = 2 
        else:
            label_int = 3
        
        label_idx = torch.LongTensor(1)
        label_idx[0] = label_int 
        label = torch.FloatTensor(self.n_lables)
        label.zero_()
        label.scatter_(0, label_idx, 1)
        
        sample = {'image_left': image_left, 
                  'image_right': image_right, 
                  'label': label.numpy()}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class MirkoDatasetReg(Dataset):
    """Face trajectories dataset."""

    def __init__(self, root_dir, training_per, transform=None, dset_type='train', seed=1, permuted= True):
        """
        Args:
            indices_file (string): Path to the txt file with image indices.
            traj_dir (string): Directory with all the trajectories.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.num_train_sets = len(root_dir)
        
        self.root_dir = root_dir
        self.labels = []
        self.indices = []
        
        for i in range(self.num_train_sets):
            
            self.labels.append(np.load(self.root_dir[i] + 'labels_abs.npy'))
            
            nframes = int(len(self.labels[i]))
    
            np.random.seed(seed)
            if permuted == True:
                permuted_indeces = np.random.permutation(range(nframes))
            else:
                permuted_indeces = range(nframes)
            
            train_number = int(nframes * training_per[i])
            if dset_type == 'train':
                self.indices.append(permuted_indeces[:train_number])
            else:
                self.indices.append(permuted_indeces[train_number:])
            
        self.transform = transform

    def __len__(self):
        
        len_tot = 0
        for i in range(self.num_train_sets):
            len_tot += len(self.indices[i])
        
        return len_tot

    def __getitem__(self, idx):
        i = 0
        sum_idx = len(self.indices[i])
        while idx >= sum_idx:
            i += 1
            sum_idx += len(self.indices[i])
        sum_idx -= len(self.indices[i])
        idx -= sum_idx
        
        img_name = str(self.indices[i][idx]) + '.png'
        
        cv_image = cv2.imread(self.root_dir[i] + 'left' + img_name)
        # grab the dimensions of the image and calculate the center
        # of the image
        (h, w) = cv_image.shape[:2]
        center = (w / 2, h / 2)
        # rotate the image by 180 degrees
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        rotated = cv2.warpAffine(cv_image, M, (w, h))
        
        image_left = np.expand_dims(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), axis=2)
        cv_image = cv2.imread(self.root_dir[i] + 'right' + img_name)
        image_right = np.expand_dims(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY), axis=2)        
        
        #plt.imshow(image_left, cmap="gray")
        #plt.show()
        
        #image_left = io.imread(self.root_dir[i] + 'left' + img_name)
        #image_right = io.imread(self.root_dir[i] + 'right' + img_name)
        
        label = np.asarray(self.labels[i][self.indices[i][idx]])
        
        sample = {'image_left': image_left, 
                  'image_right': image_right, 
                  'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
        
class MirkoDatasetRegNorm(Dataset):
    """Face trajectories dataset."""

    def __init__(self, root_dir, training_per, transform=None, dset_type='train', seed=1, permuted= True):
        """
        Args:
            indices_file (string): Path to the txt file with image indices.
            traj_dir (string): Directory with all the trajectories.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.num_train_sets = len(root_dir)
        
        self.root_dir = root_dir
        self.labels = []
        
        self.indices = []
        self.dset_type = dset_type
        
        for i in range(self.num_train_sets):
            
            self.labels.append(np.load(self.root_dir[i] + 'labels.npy'))
            
            nframes = int(len(self.labels[i]))
    
            np.random.seed(seed)
            if permuted == True:
                permuted_indeces = np.random.permutation(range(nframes))
            else:
                permuted_indeces = range(nframes)
            
            train_number = int(nframes * training_per[i])
            if dset_type == 'train':
                self.indices.append(permuted_indeces[:train_number])
            else:
                self.indices.append(permuted_indeces[train_number:])
        
        if self.dset_type == 'train':        
            for i in range(self.num_train_sets):
                #print(self.labels[i])
                temp_labels = self.labels[i]
                temp_indices = self.indices[i]
                actual_labels = temp_labels[temp_indices]
                if i == 0:
                    total_labels = actual_labels
                else:
                    total_labels = np.concatenate((total_labels, actual_labels), axis=0)
                
            self.mean = np.mean(total_labels, axis=0, keepdims=False)
            self.std = np.std(total_labels, axis=0, keepdims=False)
            np.save('mean.npy', self.mean)
            np.save('std.npy', self.std)
            np.save('training_per', training_per)
        else:
            self.mean = np.load('mean.npy')
            self.std = np.load('std.npy')
            
        self.transform = transform

    def __len__(self):
        
        len_tot = 0
        for i in range(self.num_train_sets):
            len_tot += len(self.indices[i])
        
        return len_tot

    def __getitem__(self, idx):
        i = 0
        sum_idx = len(self.indices[i])
        while idx >= sum_idx:
            i += 1
            sum_idx += len(self.indices[i])
        sum_idx -= len(self.indices[i])
        idx -= sum_idx
        
        img_name = str(self.indices[i][idx]) + '.png'
        
        cv_image = cv2.imread(self.root_dir[i] + 'left' + img_name)
        # grab the dimensions of the image and calculate the center
        # of the image
        (h, w) = cv_image.shape[:2]
        center = (w / 2, h / 2)
        # rotate the image by 180 degrees
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        rotated = cv2.warpAffine(cv_image, M, (w, h))
        
        image_left = np.expand_dims(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), axis=2)
        cv_image = cv2.imread(self.root_dir[i] + 'right' + img_name)
        image_right = np.expand_dims(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY), axis=2)        
        
        #plt.imshow(image_left, cmap="gray")
        #plt.show()
        
        #image_left = io.imread(self.root_dir[i] + 'left' + img_name)
        #image_right = io.imread(self.root_dir[i] + 'right' + img_name)
        
        label = np.asarray(self.labels[i][self.indices[i][idx]])
        label = (label - self.mean) / self.std
        
        sample = {'image_left': image_left, 
                  'image_right': image_right, 
                  'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class MirkoDatasetRegNormRam(Dataset):
    """Face trajectories dataset."""

    def __init__(self, root_dir, training_per, transform=None, dset_type='train', seed=1, permuted= True):
        """
        Args:
            indices_file (string): Path to the txt file with image indices.
            traj_dir (string): Directory with all the trajectories.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.num_train_sets = len(root_dir)
        
        self.root_dir = root_dir
        self.labels = []
        
        self.indices = []
        self.leftImages = []
        self.rightImages = []
        self.dset_type = dset_type
        
        for i in range(self.num_train_sets):
            
            # Check this changeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
            self.labels.append(np.load(self.root_dir[i] + 'labels.npy'))
            #self.labels.append(np.load(self.root_dir[i] + 'labels.npy') / 1000.0)
            
            nframes = int(len(self.labels[i]))
    
            np.random.seed(seed)
            if permuted == True:
                permuted_indeces = np.random.permutation(range(nframes))
            else:
                permuted_indeces = range(nframes)
            
            train_number = int(nframes * training_per[i])
            if dset_type == 'train':
                self.indices.append(permuted_indeces[:train_number])
            else:
                self.indices.append(permuted_indeces[train_number:])
                
            for j in self.indices[i]:
                img_name = str(j) + '.png'
                cv_image = cv2.imread(self.root_dir[i] + 'left' + img_name)
                # grab the dimensions of the image and calculate the center
                # of the image
                (h, w) = cv_image.shape[:2]
                center = (w / 2, h / 2)
                # rotate the image by 180 degrees
                M = cv2.getRotationMatrix2D(center, 180, 1.0)
                rotated = cv2.warpAffine(cv_image, M, (w, h))
                
                self.leftImages.append(np.expand_dims(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), axis=2))
                cv_image = cv2.imread(self.root_dir[i] + 'right' + img_name)
                self.rightImages.append(np.expand_dims(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY), axis=2))
        
        if self.dset_type == 'train':        
            for i in range(self.num_train_sets):
                print(self.labels[i])
                temp_labels = self.labels[i]
                temp_indices = self.indices[i]
                actual_labels = temp_labels[temp_indices]
                if i == 0:
                    total_labels = actual_labels
                else:
                    total_labels = np.concatenate((total_labels, actual_labels), axis=0)
                
            self.mean = np.mean(total_labels, axis=0, keepdims=False)
            self.std = np.std(total_labels, axis=0, keepdims=False)
            np.save('mean.npy', self.mean)
            np.save('std.npy', self.std)
            np.save('training_per', training_per)
        else:
            self.mean = np.load('mean.npy')
            self.std = np.load('std.npy')
            
        self.transform = transform

    def __len__(self):
        
        len_tot = 0
        for i in range(self.num_train_sets):
            len_tot += len(self.indices[i])
        
        return len_tot

    def __getitem__(self, idx):
        img_idx = idx
        
        i = 0
        sum_idx = len(self.indices[i])
        while idx >= sum_idx:
            i += 1
            sum_idx += len(self.indices[i])
        sum_idx -= len(self.indices[i])
        idx -= sum_idx
              
        image_left = self.leftImages[img_idx]
        image_right = self.rightImages[img_idx]        
        
        #plt.imshow(image_left, cmap="gray")
        #plt.show()
        
        #image_left = io.imread(self.root_dir[i] + 'left' + img_name)
        #image_right = io.imread(self.root_dir[i] + 'right' + img_name)
        
        label = np.asarray(self.labels[i][self.indices[i][idx]])
        label = (label - self.mean) / self.std
        
        sample = {'image_left': image_left, 
                  'image_right': image_right, 
                  'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class MirkoDatasetRegOld(Dataset):
    """Face trajectories dataset."""

    def __init__(self, root_dir, transform=None, dset_type='train', seed=1, training_per = 0.7):
        """
        Args:
            indices_file (string): Path to the txt file with image indices.
            traj_dir (string): Directory with all the trajectories.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        
        self.labels = np.load(self.root_dir + 'labels.npy')
        
        nframes = int(len(self.labels))

        np.random.seed(seed)
        permuted_indeces = np.random.permutation(range(nframes))
        
        train_number = int(nframes * training_per)
        if dset_type == 'train':
            self.indices = permuted_indeces[:train_number]
        else:
            self.indices = permuted_indeces[train_number:]
            
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_name = str(self.indices[idx]) + '.png'
        image_left = io.imread(self.root_dir + 'left' + img_name)
        image_right = io.imread(self.root_dir + 'right' + img_name)
        
        label = np.asarray(self.labels[self.indices[idx]])
        
        sample = {'image_left': image_left, 
                  'image_right': image_right, 
                  'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class MirkoDatasetFromVideo(Dataset):
    """Face trajectories dataset."""

    def __init__(self, root_dir, transform=None, dset_type='Train', seed=1):
        """
        Args:
            indices_file (string): Path to the txt file with image indices.
            traj_dir (string): Directory with all the trajectories.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.reader_left = imageio.get_reader(root_dir + 'eye0.mp4')
        self.reader_right = imageio.get_reader(root_dir + 'eye1.mp4')
        metadata = self.reader_left.get_meta_data()
        nframes = metadata['nframes']

        np.random.seed(seed)
        permuted_indeces = np.random.permutation(range(nframes))
        
        train_number = int(nframes * 0.7)
        if dset_type == 'Train':
            self.indices = permuted_indeces[:train_number]
        else:
            self.indices = permuted_indeces[train_number:]
            
        

        self.fps = metadata['fps']
        self.root_dir = root_dir
        self.transform = transform
        self.n_lables = 4

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_left = self.reader_left.get_data(self.indices[idx])
        image_right = self.reader_right.get_data(self.indices[idx])
        
        
        label_idx = torch.LongTensor(1)
        label_int = int(self.indices[idx] / (self.fps * 4))
        label_idx[0] = label_int  
        label = torch.FloatTensor(self.n_lables)
        label.zero_()
        label.scatter_(0, label_idx, 1)
        
        sample = {'image_left': image_left, 
                  'image_right': image_right, 
                  'label': label.numpy()}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
#reader = imageio.get_reader('data/eye1.mp4')
#metadata = reader.get_meta_data()
#nframes = metadata['nframes']
#fps = metadata['fps']
#
#np.random.seed(2)
#permuted_indeces = np.random.permutation(range(nframes))
#
#print(permuted_indeces)
#
#np.random.seed(2)
#permuted_indeces = np.random.permutation(range(nframes))
#
#print(permuted_indeces)
#
#np.random.seed(5)
#permuted_indeces = np.random.permutation(range(nframes))
#
#print(permuted_indeces)
#
#train_number = int(nframes * 0.7)
#
#train_idx = permuted_indeces[:train_number]
#val_idx = permuted_indeces[train_number:]
#
#train_idx.size
#
#image = reader.get_data(10)
#
#
#plt.imshow(image)
#
#plt.show()
#
#print (str(train_idx.size) + ' ' + str(val_idx.size))
#
#raw_input('Press enter to continue: ')
#
#    

#data_transforms_custom1 = transforms.Compose([Rescale((120, 160)),
#                                              RandomNoise(var=0.01)])
#
#dataset_train = MirkoDatasetReg(root_dir = 'dataRegression/',
#                             transform = data_transforms_custom1,
#                             dset_type='train', seed=3)
#
#sample = dataset_train[3]
#
#print(sample['label'])
#
#image = sample['image_left']
#
#plt.imshow(image)

#plt.figure()

#sample = dataset_train[3]
#
#image = sample['image_left']
#
#plt.imshow(image)
#
#plt.show()

#raw_input('Press enter to continue: ')