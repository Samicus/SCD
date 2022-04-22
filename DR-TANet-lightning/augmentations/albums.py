
import cv2
from PIL import Image
import os
from os.path import join as pjoin, splitext as spt
import numpy as np
import albumentations as A

"""
Sharpen
Superpixel?
"""

if __name__=="__main__":

    # alpha=(0.4, 0.5)
    #lightness=(0.5, 1.0)
    transform = A.Compose([
    #A.Sharpen(p=1),

    A.Sharpen (alpha=(.5, .5), lightness=(1,1), p=1) # on both
    #A.RandomBrightnessContrast(p=0.2),
    ])
    dirname = os.path.dirname
    root = '/home/samnehme/Dev/SCD_project/TSUNAMI/set0/train'
    img_t0_root = pjoin(root,'t0')
    img_t1_root = pjoin(root,'t1')
    img_mask_root = pjoin(root,'mask/bmp')
    filename = list(spt(f)[0] for f in os.listdir(img_mask_root))
    
    index = 25
    fn = filename[index]
    fn_t0 = pjoin(img_t0_root,fn+'.jpg')
    fn_t1 = pjoin(img_t1_root,fn+'.jpg')
    fn_mask = pjoin(img_mask_root,fn+'.bmp')
    
    t0 = cv2.imread(fn_t0, 1)
    t1 = cv2.imread(fn_t1, 1)
    mask = cv2.imread(fn_mask, 0)


    #t0 = cv2.imread(fn_t0, 1)
    #t1 = cv2.imread(fn_t1, 1)
    #mask = cv2.imread(fn_mask, 0)

    index = 11

    pillow_image = Image.open(fn_t0)
    image = np.array(pillow_image)
    transformed = transform(image=image)["image"]

    img_t0 = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
    img_t0 = Image.fromarray(img_t0.astype(np.uint8))
    img_t0.show()

    pillow_image = Image.open(fn_t1)
    image = np.array(pillow_image)
    transformed = transform(image=image)["image"]

    img_t1 = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
    img_t1 = Image.fromarray(img_t1.astype(np.uint8))
    img_t1.show()
    
    img_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    img_mask = Image.fromarray(img_mask.astype(np.uint8))
    img_mask.show()
    