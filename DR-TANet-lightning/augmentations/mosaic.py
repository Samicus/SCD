import random
import numpy as np
import cv2
import math
import os
from os.path import join as pjoin, splitext as spt
from PIL import Image
from params import scale, translate

def mosaic_augment(index, filename, t0_root, t1_root, mask_root, img_shape=(224,1024)):
    """
    Description:
        - This method augments an image using the mosaic augmentation method. It combines 4 images 
          by translating and rescaling without changing the aspect ratio. Additional 
          augmentations such as rotation, shear and perspective can be applied. The hyper parameters
          for the augmentations can be found in params.py.

    PARAMS: 
        -index: Index of the current image
        -filename:  list of image filenames
        -t0_root:   root directory to t0 images
        -t1_root:   root directory to t1 images
        -mask_root: root directory to mask
    Output:
        - img4_t0, img4_t1, img4_mask:
                The output is the augmented images t0, t1 and mask as 
                numpy ndarrays with the same output size as the original. 
        
    """
    nr_images = len(filename)
    indices = range(nr_images-1)        

    #fn1 = filename[index]
    #fn1_t0 = pjoin(t0_root,fn1+'.jpg')
    #img1_t0 = cv2.imread(fn1_t0, 1)  # Only need the shape add as argument to the method, with a default value
    s = img_shape[0]            # Only need the shape 
    s1 = img_shape[1]
    
    mosaic_border = [-s//2 , -s1//2]
    #yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in mosaic_border)  # mosaic center x, y
    #yc = int(random.uniform(-mosaic_border[0],  s + mosaic_border[0]))
    #xc = int(random.uniform(-mosaic_border[1],  s1 + mosaic_border[1]))
    yc = 224  #int(random.uniform(-20,  s + mosaic_border[0]))
    xc = 1024 #int(1024/2)
    
    #yc, xc = (int(random.uniform(-x,  x)) for x in mosaic_border)
    indices = [index] + random.choices(indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    #print(indices)
    for i, index in enumerate(indices):
        # Load image
        
        fn = filename[index]

        fn_t0 =   pjoin(t0_root,fn+'.jpg')
        fn_t1 =   pjoin(t1_root,fn+'.jpg')
        fn_mask = pjoin(mask_root,fn+'.bmp')

        img_t0 =   cv2.imread(fn_t0, 1)
        img_t1 =   cv2.imread(fn_t1, 1)
        img_mask = cv2.imread(fn_mask, 0)

        # scale each image seperately
        resize_factor = random.uniform(scale[0], scale[1])
        h_orig, w_orig, _ = img_t0.shape
        img_t0 =   cv2.resize(img_t0,   (int(resize_factor*w_orig), int(resize_factor*h_orig)))
        img_t1 =   cv2.resize(img_t1,   (int(resize_factor*w_orig), int(resize_factor*h_orig)))
        img_mask = cv2.resize(img_mask, (int(resize_factor*w_orig), int(resize_factor*h_orig)))
 
        h, w, _ = img_t0.shape
        
        # place img in img4
        if i == 0:  # top left
            img4_t0 = np.full((s * 2, s1 * 2, img_t0.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            img4_t1 = np.full((s * 2, s1 * 2, img_t1.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            img4_mask = np.full((s * 2, s1 * 2), 255, dtype=np.uint8)                   # base image with 4 tiles


            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s1 * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s1 * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4_t0[y1a:y2a, x1a:x2a] = img_t0[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        img4_t1[y1a:y2a, x1a:x2a] = img_t1[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        img4_mask[y1a:y2a, x1a:x2a] = img_mask[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

    
    img4_t0, img4_t1, img4_mask = random_perspective(img4_t0, img4_t1, img4_mask, translate=translate,  border=mosaic_border)#, border=mosaic_border)  # border to remove
    
    
    return img4_t0, img4_t1, img4_mask


def random_perspective(im_t0, im_t1, im_mask, degrees=0, translate=.2, scale=0, shear=0, perspective=0.0,
                       border=(0, 0)):
    


    height = im_t0.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im_t0.shape[1] + border[1] * 2
   
    # Center
    C = np.eye(3)
    C[0, 2] = -im_t0.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im_t0.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im_t0 = cv2.warpPerspective(im_t0, M, dsize=(width, height), borderValue=(114, 114, 114))
            im_t1 = cv2.warpPerspective(im_t1, M, dsize=(width, height), borderValue=(114, 114, 114))
            im_mask = cv2.warpPerspective(im_mask, M, dsize=(width, height), borderValue=(255, 255, 255))
        else:  # affine
            im_t0 = cv2.warpAffine(im_t0, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            im_t1 = cv2.warpAffine(im_t1, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            im_mask = cv2.warpAffine(im_mask, M[:2], dsize=(width, height), borderValue=(255, 255, 255))

    return im_t0, im_t1, im_mask


if __name__ == '__main__':

    dirname = os.path.dirname
    root = pjoin(dirname(dirname(dirname(dirname(__file__)))), "TSUNAMI/set0/train")
    img_t0_root = pjoin(root,'t0')
    img_t1_root = pjoin(root,'t1')
    img_mask_root = pjoin(root,'mask/bmp')
    filename = list(spt(f)[0] for f in os.listdir(img_mask_root))
    
    index = 11
    fn = filename[index]
    fn_t0 = pjoin(img_t0_root,fn+'.jpg')
    fn_t1 = pjoin(img_t1_root,fn+'.jpg')
    fn_mask = pjoin(img_mask_root,fn+'.bmp')
    
    t0 = cv2.imread(fn_t0, 1)
    t1 = cv2.imread(fn_t1, 1)
    mask = cv2.imread(fn_mask, 0)

    index
    img_t0, img_t1, img_mask = mosaic_augment(index, filename, img_t0_root, img_t1_root, img_mask_root)

    img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
    img_t0 = Image.fromarray(img_t0.astype(np.uint8))
    img_t0.show()

    img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
    img_t1 = Image.fromarray(img_t1.astype(np.uint8))
    img_t1.show()

    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
    img_mask = Image.fromarray(img_mask.astype(np.uint8))
    img_mask.show()