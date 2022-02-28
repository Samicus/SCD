import cv2
import numpy as np

"""
Variant of rotate_pcd.py that can be used during training.
"""

def rotate_augment(img_t0, img_t1, mask):
    h, w = mask.shape
    center = (w // 2, h // 2)
    angle = np.random.randint(0, 360)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_t0_rot = cv2.warpAffine(img_t0, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    img_t1_rot = cv2.warpAffine(img_t1, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask_rot = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return img_t0_rot, img_t1_rot, mask_rot