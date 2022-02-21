import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join as pjoin, splitext as spt
from params import mosaic_aug
from mosaic import mosaic_augment

def check_validness(f):
    return any([i in spt(f)[1] for i in ['jpg','bmp']])

class PCD(Dataset):

    def __init__(self, root):
        super(PCD, self).__init__()
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
        
        if mosaic_aug:
            img_t0, img_t1, mask = mosaic_augment(index, self.filename, self.img_t0_root, self.img_t1_root, self.img_mask_root, img_shape=(224, 1024))
        else:
            img_t0 = cv2.imread(fn_t0, 1)
            img_t1 = cv2.imread(fn_t1, 1)
        
        # Invert BMP mask
        mask = 255 - cv2.imread(fn_mask, 0)
        
        w, h, c = img_t0.shape
        r = 288. / min(w, h)
        # resize images so that min(w, h) == 256
        img_t0_r = cv2.resize(img_t0, (int(r * w), int(r * h)))
        img_t1_r = cv2.resize(img_t1, (int(r * w), int(r * h)))
        mask_r = cv2.resize(mask, (int(r * w), int(r * h)))[:, :, np.newaxis]
        
        img_t0 = np.asarray(img_t0_r).astype('f').transpose(2, 0, 1) / 128.0 - 1.0
        img_t1 = np.asarray(img_t1_r).astype('f').transpose(2, 0, 1) / 128.0 - 1.0
        mask = np.asarray(mask_r>128).astype('f').transpose(2, 0, 1)

        input_ = torch.from_numpy(np.concatenate((img_t0, img_t1), axis=0))
        mask_ = torch.from_numpy(mask)#.long()
        
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

        w, h, c = img_t0.shape
        w_r = int(256*max(w/256,1))
        h_r = int(256*max(h/256,1))
        # resize images so that min(w, h) == 256
        img_t0_r = cv2.resize(img_t0,(h_r,w_r))
        img_t1_r = cv2.resize(img_t1,(h_r,w_r))
        mask_r = cv2.resize(mask,(h_r,w_r))[:, :, np.newaxis]
        
        img_t0_r = np.asarray(img_t0_r).astype('f').transpose(2, 0, 1) / 128.0 - 1.0
        img_t1_r = np.asarray(img_t1_r).astype('f').transpose(2, 0, 1) / 128.0 - 1.0
        mask_r = np.asarray(mask_r > 128).astype('f').transpose(2, 0, 1)

        return img_t0_r, img_t1_r, mask_r, w, h, w_r, h_r

    def __len__(self):
        return len(self.filename)

    def get_random_image(self):
        idx = np.random.randint(0,len(self))
        return self.__getitem__(idx)