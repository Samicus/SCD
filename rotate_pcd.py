# import the necessary packages
from re import M
import numpy as np
import argparse
import imutils
import cv2
from os.path import join as pjoin
from PIL import Image
import glob

""" Required dataset structure

Only training data is rotated

PCD        
├── train/
│   ├── mask/       # *.png
|       ├── bmp/    # *.bmp
│   ├── t0/         # *.jpg
|   ├── t1/         # *.jpg

Usage:

python3 rotate_pcd.py -i /path/to/dataset
    
    optional flags for rotating specific data
        --mask
        --bmp
        --t0
        --t1

"""

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--dataset", required=True,
	help="path to the dataset")
parser.add_argument("--mask", action="store_true")
parser.add_argument("--bmp", action="store_true")
parser.add_argument("--t0", action="store_true")
parser.add_argument("--t1", action="store_true")
args = vars(parser.parse_args())

pcd_path = args["dataset"]
DEGREE_INCREMENT = 6
TRAIN_PATH = pjoin(pcd_path, "train")

mask_path = pjoin(TRAIN_PATH, "mask")
bmp_path = pjoin(mask_path, "bmp")
t0_path = pjoin(TRAIN_PATH, "t0")
t1_path = pjoin(TRAIN_PATH, "t1")

def rotate_mask():
    for filepath in glob.glob(pjoin(mask_path, '*.png')):
        image = cv2.imread(filepath)
        for angle in np.arange(0, 360, DEGREE_INCREMENT):
            rotated = imutils.rotate(image, angle)
            filename = filepath.split('/')[-1]  # xxxx.png
            filename = filename.split('.')[0]   # xxxx
            filename += "_{}.png".format(angle) # xxxx_angle.png
            cv2.imwrite(pjoin(mask_path, filename), rotated)

def rotate_bmp():
    for filepath in glob.glob(pjoin(bmp_path, '*.bmp')):
        image = cv2.imread(filepath)
        for angle in np.arange(0, 360, DEGREE_INCREMENT):
            rotated = imutils.rotate(image, angle)
            filename = filepath.split('/')[-1]  # xxxx.bmp
            filename = filename.split('.')[0]   # xxxx
            filename += "_{}.bmp".format(angle) # xxxx_angle.bmp
            cv2.imwrite(pjoin(bmp_path, filename), rotated)

def rotate_t0():
    for filepath in glob.glob(pjoin(t0_path, '*.jpg')):
        image = cv2.imread(filepath)
        for angle in np.arange(0, 360, DEGREE_INCREMENT):
            rotated = imutils.rotate(image, angle)
            filename = filepath.split('/')[-1]  # xxxx.jpg
            filename = filename.split('.')[0]   # xxxx
            filename += "_{}.jpg".format(angle) # xxxx_angle.jpg
            cv2.imwrite(pjoin(t0_path, filename), rotated)

def rotate_t1():
    for filepath in glob.glob(pjoin(t1_path, '*.jpg')):
        image = cv2.imread(filepath)
        for angle in np.arange(0, 360, DEGREE_INCREMENT):
            rotated = imutils.rotate(image, angle)
            filename = filepath.split('/')[-1]  # xxxx.jpg
            filename = filename.split('.')[0]   # xxxx
            filename += "_{}.jpg".format(angle) # xxxx_angle.jpg
            cv2.imwrite(pjoin(t1_path, filename), rotated)

if args["mask"]:
    rotate_mask()
if args["bmp"]:
    rotate_bmp()
if args["t0"]:
    rotate_t0()
if args["t1"]:
    rotate_t1()