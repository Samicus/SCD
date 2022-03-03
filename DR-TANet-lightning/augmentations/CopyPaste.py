from PIL import ImageFilter, Image
import random
from torchvision import transforms
import numpy as np
import cv2
import skimage.exposure
from scipy.ndimage import gaussian_filter
import pickle
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from os.path import join as pjoin, splitext as spt
from PIL import Image


class CopyPaste(object):
    ''' Copy paste augumentation:
        params: 
    '''


    def __init__(self, t0_root, t1_root, mask_root, filename, sigma=1, scale=[0.3,0.8], rotation=180):
        self.sigma = sigma
        self.scale = scale
        self.t0_root = t0_root
        self.t1_root = t1_root
        self.mask_root = mask_root
        self.filename = filename
        self.rotation = rotation
    def __call__(self, img_t0, img_t1, img_mask):
        copy_t0, copy_t1, copy_mask = self.load_random_imgs()
    

        h, w, _ = img_t0.shape

        #copy_t0 = transforms.Resize(size=(H, W))(self.instance)
        resize_factor = random.uniform(self.scale[0], self.scale[1])
        copy_t0 =   cv2.resize(copy_t0,   (int(resize_factor*w), int(resize_factor*h)))
        copy_t1 =   cv2.resize(copy_t1,   (int(resize_factor*w), int(resize_factor*h)))
        copy_mask = cv2.resize(copy_mask, (int(resize_factor*w), int(resize_factor*h)))
        random_rotation = random.uniform(-self.rotation, self.rotation)
        copy_t0 = self.rotate_image(copy_t0, random_rotation)
        copy_t1 = self.rotate_image(copy_t1, random_rotation)
        
        copy_mask = self.rotate_image(copy_mask, random_rotation)
        h_resized, w_resized, _ = copy_t0.shape

        #Convert to np arrays
        copy_t0 = np.asarray(copy_t0)
        copy_t1 = np.asarray(copy_t1)
        copy_mask = np.asarray(copy_mask)

        img_t0 = np.asarray(img_t0)
        img_t1 = np.asarray(img_t1)
        img_mask = np.asarray(img_mask)

        h_start = np.random.randint(1,h-h_resized-1)
        w_start = np.random.randint(1, w-w_resized-1)

        y1, y2 = h_start, h_start + h_resized
        x1, x2 = w_start, w_start + w_resized

        binary_mask = 1.0 * (copy_mask > 0)
        # blur_binary_mask = skimage.exposure.rescale_intensity(blur_binary_mask)
        invert_mask = 1.0 * (np.logical_not(binary_mask).astype(int))

        blur_invert_mask = np.expand_dims(invert_mask, 2)  # Expanding dims to match channels
        blur_binary_mask = np.expand_dims(binary_mask, 2)

        img_t0[y1:y2, x1:x2] = (copy_t0 * blur_invert_mask) + (img_t0[y1:y2, x1:x2] * blur_binary_mask)
        img_t1[y1:y2, x1:x2] = (copy_t1 * blur_invert_mask) + (img_t1[y1:y2, x1:x2] * blur_binary_mask)
        img_mask[y1:y2, x1:x2] = (copy_mask * invert_mask) + (img_mask[y1:y2, x1:x2] * binary_mask)

        return img_t0, img_t1, img_mask


    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rotated_img = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        return rotated_img

    def load_random_imgs(self):
        """
        loads current images t0, t1 and mask into lists with size 1
        """
        nr_images = len(self.filename)
        index = np.random.randint(0, nr_images) 
        fn = self.filename[index]

        fn_t0 =   pjoin(self.t0_root,fn+'.jpg')
        fn_t1 =   pjoin(self.t1_root,fn+'.jpg')
        fn_mask = pjoin(self.mask_root,fn+'.bmp')

        img_t0 =   cv2.imread(fn_t0, 1)
        img_t1 =   cv2.imread(fn_t1, 1)
        img_mask = cv2.imread(fn_mask, 0)
        #img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)
        return img_t0, img_t1, img_mask
        

if __name__ == "__main__":
    dirname = os.path.dirname
    #root = pjoin(dirname(dirname(dirname(dirname(__file__)))), "TSUNAMI/set0/train")
    root = '/home/samnehme/Dev/SCD_project/TSUNAMI/set0/train'
    img_t0_root = pjoin(root,'t0')
    img_t1_root = pjoin(root,'t1')
    img_mask_root = pjoin(root,'mask/bmp')
    filename = list(spt(f)[0] for f in os.listdir(img_mask_root))
    copy_paste =  CopyPaste(img_t0_root, img_t1_root, img_mask_root, filename, sigma=1, scale=[0.1,1])
    
    index = 11
    fn = filename[index]
    fn_t0 = pjoin(img_t0_root,fn+'.jpg')
    fn_t1 = pjoin(img_t1_root,fn+'.jpg')
    fn_mask = pjoin(img_mask_root,fn+'.bmp')
    
    t0 = cv2.imread(fn_t0, 1)
    t1 = cv2.imread(fn_t1, 1)
    mask = cv2.imread(fn_mask, 0)
    img_t0, img_t1, img_mask = copy_paste(t0, t1, mask)
    
    img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
    img_t0 = Image.fromarray(img_t0.astype(np.uint8))
    img_t0.show()

    
    img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
    img_t1 = Image.fromarray(img_t1.astype(np.uint8))
    img_t1.show()
    
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
    img_mask = Image.fromarray(img_mask.astype(np.uint8))
    img_mask.show()
    