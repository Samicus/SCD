import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join as pjoin, splitext as spt
from params import augment_on
from data.augmentations import DataAugment


def check_validness(f):
    return any([i in spt(f)[1] for i in ['jpg','bmp']])

class PCDfull(Dataset):

    def __init__(self, root):
        super(PCDfull, self).__init__()
        self.img_t0_root = pjoin(root,'t0')
        self.img_t1_root = pjoin(root,'t1')
        self.img_mask_root = pjoin(root,'mask')
        self.filename = list(spt(f)[0] for f in os.listdir(self.img_mask_root) if check_validness(f))
        self.filename.sort()
        self.data_augment = DataAugment(self.img_t0_root, self.img_t1_root, self.img_mask_root, self.filename )
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
        
        # TODO: Integrate mosaic_aug and random_erase_aug into a main augmentation script
        if augment_on:
            img_t0, img_t1, mask = self.data_augment(index)
        else:
            img_t0 = cv2.imread(fn_t0, 1)
            img_t1 = cv2.imread(fn_t1, 1)
            mask = cv2.imread(fn_mask, 0)

        # Invert BMP mask
        mask = 255 - mask

        img_t0_r_ = np.asarray(img_t0).astype('f').transpose(2, 0, 1) / 255.0               # -- > (RGB, height, width)
        img_t1_r_ = np.asarray(img_t1).astype('f').transpose(2, 0, 1) / 255.0               # -- > (RGB, height, width)
        mask_r_ = np.asarray(mask[:, :, np.newaxis]>128).astype('f').transpose(2, 0, 1)     # -- > (RGB, height, width)

        input_ = np.concatenate((img_t0, img_t1))
        
        return input_, mask_

    def __len__(self):
        return len(self.filename)

    def get_random_image(self):
        idx = np.random.randint(0,len(self))
        return self.__getitem__(idx)


class PCDeval(Dataset):

    def __init__(self, root):
        super(PCDeval, self).__init__()
        self.img_t0_root = pjoin(root, 't0')
        self.img_t1_root = pjoin(root, 't1')
        self.img_mask_root = pjoin(root, 'mask')
        self.filename = list(spt(f)[0] for f in os.listdir(self.img_mask_root) if check_validness(f))
        self.filename.sort()

    def __getitem__(self, index):

        fn = self.filename[index]
        fn_t0 = pjoin(self.img_t0_root, fn + '.jpg')
        fn_t1 = pjoin(self.img_t1_root, fn + '.jpg')
        fn_mask = pjoin(self.img_mask_root, fn + '.bmp')

        if os.path.isfile(fn_t0) == False:
            print('Error: File Not Found: ' + fn_t0)
            exit(-1)
        if os.path.isfile(fn_t1) == False:
            print('Error: File Not Found: ' + fn_t1)
            exit(-1)
        if os.path.isfile(fn_mask) == False:
            print('Error: File Not Found: ' + fn_mask)
            exit(-1)

        img_t0 = cv2.imread(fn_t0, 1)
        img_t1 = cv2.imread(fn_t1, 1)
        
        # Invert BMP mask
        mask = 255 - cv2.imread(fn_mask, 0)

        img_t0_r_ = np.asarray(img_t0).astype('f').transpose(2, 0, 1)                       # -- > (RGB, height, width)
        img_t1_r_ = np.asarray(img_t1).astype('f').transpose(2, 0, 1)                       # -- > (RGB, height, width)
        mask_r_ = np.asarray(mask[:, :, np.newaxis]>128).astype('f').transpose(2, 0, 1)     # -- > (RGB, height, width)
        
        input_r_ = np.concatenate((img_t0_r_, img_t1_r_))

        return input_r_, mask_r_

    def __len__(self):
        return len(self.filename)

    def get_random_image(self):
        idx = np.random.randint(0,len(self))
        return self.__getitem__(idx)
    
class PCDcrop(Dataset):

    def __init__(self, root):
        super(PCDcrop, self).__init__()
        self.img_t0_root = pjoin(root,'t0')
        self.img_t1_root = pjoin(root,'t1')
        self.img_mask_root = pjoin(root,'mask')
        self.filename = list(spt(f)[0] for f in os.listdir(self.img_mask_root) if check_validness(f))
        self.filename.sort()
        
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
        
        img_t0 = cv2.imread(fn_t0, 1)
        img_t1 = cv2.imread(fn_t1, 1)
        
        # Invert BMP mask
        mask = 255 - cv2.imread(fn_mask, 0)
                
        img_t0_r_ = np.asarray(img_t0).astype('f').transpose(2, 0, 1) / 255.0               # -- > (RGB, height, width)
        img_t1_r_ = np.asarray(img_t1).astype('f').transpose(2, 0, 1) / 255.0               # -- > (RGB, height, width)
        mask_r_ = np.asarray(mask[:, :, np.newaxis]>128).astype('f').transpose(2, 0, 1)     # -- > (RGB, height, width)

        crop_width = 256
        _, h, w = img_t0_r_.shape
        try:
            x_l = np.random.randint(0, w - crop_width)
        except ValueError:
            x_l = 0
        x_r = x_l + crop_width
        try:
            y_l = np.random.randint(0, h - crop_width)
        except ValueError: y_l = 0
        y_r = y_l + crop_width

        input_ = np.concatenate((img_t0_r_[:, y_l:y_r, x_l:x_r], img_t1_r_[:, y_l:y_r, x_l:x_r]), axis=0)
        mask_ = mask_r_[:, y_l:y_r, x_l:x_r]
        
        return input_, mask_

    def __len__(self):
        return len(self.filename)

    def get_random_image(self):
        idx = np.random.randint(0,len(self))
        return self.__getitem__(idx)
