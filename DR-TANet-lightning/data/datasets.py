import os
from random import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from os.path import join as pjoin, splitext as spt
from data.augmentations import DataAugment
import albumentations as A
from PIL import Image
from util import load_config
import torch

def check_validness(f):
    return any([i in spt(f)[1] for i in ['jpg', 'bmp', 'png']])

class PCD(Dataset):

    def __init__(self, root, augmentations, AUGMENT_ON, PCD_CONFIG, PCD_FRACTION, trial=None):
        super(PCD, self).__init__()
        
        self.img_t0_root = pjoin(root,'t0')
        self.img_t1_root = pjoin(root,'t1')
        self.img_mask_root = pjoin(root,'mask')
        self.filename = list(spt(f)[0] for f in os.listdir(self.img_mask_root) if check_validness(f))
        self.filename = self.filename[:int(PCD_FRACTION * len(self.filename))]  # Limit to percentage of list
        self.filename.sort()
        aug_params  = load_config("DR-TANet-lightning/config/augparams.yaml")
        self.AUGMENT_ON = AUGMENT_ON

        self.data_augment = DataAugment(self.img_t0_root, self.img_t1_root, self.img_mask_root, self.filename, augmentations, aug_params, shape=(224, 1024), trial=trial)    # (height, width)
        self.PCD_CONFIG = PCD_CONFIG
        
        self.transform = A.Compose([
                A.Sharpen (alpha=(1, 1), lightness=(0,0), p=1)
                ])
    def __getitem__(self, index):
        
        fn = self.filename[index]
        fn_t0 = pjoin(self.img_t0_root, fn+'.jpg')
        fn_t1 = pjoin(self.img_t1_root, fn+'.jpg')
        fn_mask = pjoin(self.img_mask_root, fn+'.bmp')

        if os.path.isfile(fn_t0) == False:
            print('Error: File Not Found: ' + fn_t0)
            exit(-1)
        if os.path.isfile(fn_t1) == False:
            print('Error: File Not Found: ' + fn_t1)
            exit(-1)
        if os.path.isfile(fn_mask) == False:
            print('Error: File Not Found: ' + fn_mask)
            exit(-1)

        # Augmentations
        if self.AUGMENT_ON:
            img_t0, img_t1, mask = self.data_augment(index)
        else:
            img_t0 = cv2.imread(fn_t0, 1)
            img_t1 = cv2.imread(fn_t1, 1)
            mask = cv2.imread(fn_mask, 0)
        
        # Invert BMP mask
        mask = 255 - mask
        
        # Normalization
        img_t0_r_ = np.asarray(img_t0).astype('f').transpose(2, 1, 0) / 255.0               # -- > (RGB, height, width)
        img_t1_r_ = np.asarray(img_t1).astype('f').transpose(2, 1, 0) / 255.0               # -- > (RGB, height, width)
        mask_r_ = np.asarray(mask[:, :, np.newaxis]>128).astype('f').transpose(2, 1, 0)     # -- > (RGB, height, width)
        
        # Cropped or full images
        if self.PCD_CONFIG == "crop":
            input_, mask_ = self.crop(img_t0_r_, img_t1_r_, mask_r_)
        elif self.PCD_CONFIG == "full":
            input_ = np.concatenate((img_t0_r_, img_t1_r_))
            mask_ = mask_r_
        else:
            print("Choose a valid PCD config")
            exit(1)
        
        return input_, mask_

    def __len__(self):
        return len(self.filename)

    def get_random_image(self):
        idx = np.random.randint(0,len(self))
        return self.__getitem__(idx)
    
    def crop(self, img_t0, img_t1, mask):
        
        _, h, w = img_t0.shape
        crop_size = 224
        
        try:
            x_l = np.random.randint(0, w - crop_size)
        except ValueError:
            x_l = 0
        x_r = x_l + crop_size
        try:
            y_l = np.random.randint(0, h - crop_size)
        except ValueError:
            y_l = 0
        y_r = y_l + crop_size
        
        # Random crop
        img_t0 = torch.from_numpy(img_t0[:, y_l:y_r, x_l:x_r])
        img_t1 = torch.from_numpy(img_t1[:, y_l:y_r, x_l:x_r])
        mask = torch.from_numpy(mask[:, y_l:y_r, x_l:x_r])
        
        # Random rotation
        num_rot = np.random.randint(0, 4)
        img_t0 = torch.rot90(img_t0, k=num_rot, dims=(1, 2))
        img_t1 = torch.rot90(img_t1, k=num_rot, dims=(1, 2))
        mask = torch.rot90(mask, k=num_rot, dims=(1, 2))
        
        input_ = np.concatenate((img_t0, img_t1), axis=0)
        mask_ = mask
        
        return input_, mask_


class VL_CMU_CD(Dataset):

    def __init__(self, root, augmentations, AUGMENT_ON, trial=None):
        super(VL_CMU_CD, self).__init__()
        self.img_t0_root = pjoin(root, 't0')
        self.img_t1_root = pjoin(root, 't1')
        self.img_mask_root = pjoin(root, 'mask')
        self.filename = list(spt(f)[0] for f in os.listdir(self.img_mask_root) if check_validness(f))
        self.filename.sort()
        
        self.AUGMENT_ON = AUGMENT_ON
        self.data_augment = DataAugment(self.img_t0_root, self.img_t1_root, self.img_mask_root, self.filename, augmentations, shape=(512, 512), trial=trial)

    def __getitem__(self, index):

        fn = self.filename[index]
        fn_t0 = pjoin(self.img_t0_root, fn + '.png')
        fn_t1 = pjoin(self.img_t1_root, fn + '.png')
        fn_mask = pjoin(self.img_mask_root, fn + '.png')

        if os.path.isfile(fn_t0) == False:
            print('Error: File Not Found: ' + fn_t0)
            exit(-1)
        if os.path.isfile(fn_t1) == False:
            print('Error: File Not Found: ' + fn_t1)
            exit(-1)
        if os.path.isfile(fn_mask) == False:
            print('Error: File Not Found: ' + fn_mask)
            exit(-1)

        # Augmentations
        if self.AUGMENT_ON:
            img_t0, img_t1, mask = self.data_augment(index)
        else:
            img_t0 = cv2.imread(fn_t0, 1)
            img_t1 = cv2.imread(fn_t1, 1)
            mask = cv2.imread(fn_mask, 0)
        
        mask = 255 - cv2.imread(fn_mask, 0)

        # Normalization
        img_t0_r = np.asarray(img_t0).astype('f').transpose(2, 1, 0) / 255.0               # -- > (RGB, height, width)
        img_t1_r = np.asarray(img_t1).astype('f').transpose(2, 1, 0) / 255.0               # -- > (RGB, height, width)
        mask_r = np.asarray(mask[:, :, np.newaxis]>0).astype('f').transpose(2, 1, 0)     # -- > (RGB, height, width)
        
        input_ = np.concatenate((img_t0_r, img_t1_r))
        mask_ = mask_r
        
        return input_, mask_

    def __len__(self):
        return len(self.filename)

    def get_random_image(self):
        idx = np.random.randint(0, len(self))
        return self.__getitem__(idx)