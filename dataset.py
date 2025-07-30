# dataset.py

import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random

# Precise Min-Max scaling values based on dataset analysis
LR_MIN = 0.0206
LR_MAX = 0.2737
HR_MIN = 0.0191
HR_MAX = 0.4274

class Oli2MSIDataset(Dataset):
    def __init__(self, lr_files, hr_files, hr_crop_size, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.hr_crop_size = hr_crop_size
        self.lr_crop_size = hr_crop_size // upscale_factor
        self.lr_files = lr_files
        self.hr_files = hr_files
        assert len(self.lr_files) == len(self.hr_files)

    def __getitem__(self, index):
        lr_file = self.lr_files[index]
        hr_file = self.hr_files[index]

        with rasterio.open(lr_file) as src:
            lr_image = src.read().astype(np.float32)
        with rasterio.open(hr_file) as src:
            hr_image = src.read().astype(np.float32)

        # Normalize LR and HR images to [0, 1] first
        lr_image = (lr_image - LR_MIN) / (LR_MAX - LR_MIN)
        hr_image = (hr_image - HR_MIN) / (HR_MAX - HR_MIN)
        
        # Then normalize to [-1, 1] to work with Tanh activation
        lr_image = (lr_image * 2) - 1
        hr_image = (hr_image * 2) - 1

        # Synchronized Random Cropping
        lr_h, lr_w = lr_image.shape[1], lr_image.shape[2]
        rand_x_lr = random.randint(0, lr_w - self.lr_crop_size)
        rand_y_lr = random.randint(0, lr_h - self.lr_crop_size)
        lr_cropped = lr_image[:, rand_y_lr:rand_y_lr + self.lr_crop_size, rand_x_lr:rand_x_lr + self.lr_crop_size]
        
        rand_x_hr = rand_x_lr * self.upscale_factor
        rand_y_hr = rand_y_lr * self.upscale_factor
        hr_cropped = hr_image[:, rand_y_hr:rand_y_hr + self.hr_crop_size, rand_x_hr:rand_x_hr + self.hr_crop_size]

        # Augmentations
        if random.random() > 0.5:
            lr_cropped = np.ascontiguousarray(np.flip(lr_cropped, axis=2))
            hr_cropped = np.ascontiguousarray(np.flip(hr_cropped, axis=2))
        if random.random() > 0.5:
            lr_cropped = np.ascontiguousarray(np.flip(lr_cropped, axis=1))
            hr_cropped = np.ascontiguousarray(np.flip(hr_cropped, axis=1))
            
        return torch.from_numpy(lr_cropped), torch.from_numpy(hr_cropped), hr_file
    
    def __len__(self):
        return len(self.lr_files)