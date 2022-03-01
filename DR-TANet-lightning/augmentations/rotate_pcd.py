# import the necessary packages
from re import M
import numpy as np
import argparse
import imutils
import cv2
from os.path import join as pjoin
from PIL import Image
import glob
import os

""" Required dataset structure

Only training data is rotated

PCD        
├── train/
│   ├── mask/       # *.bmp
│   ├── t0/         # *.jpg
|   ├── t1/         # *.jpg

Usage:

python3 rotate_pcd.py -i /path/to/dataset
    
    optional flags for rotating specific data
        --mask
        --t0
        --t1

"""

DEGREE_INCREMENT = 24
NUM_CROPS = 4

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--dataset", required=True,
	help="path to the dataset")
parser.add_argument("--mask", action="store_true")
parser.add_argument("--t0", action="store_true")
parser.add_argument("--t1", action="store_true")
args = vars(parser.parse_args())

PCD_PATH = args["dataset"]
TRAIN_PATH = pjoin(PCD_PATH, "train")

mask_path = pjoin(TRAIN_PATH, "mask")
t0_path = pjoin(TRAIN_PATH, "t0")
t1_path = pjoin(TRAIN_PATH, "t1")

operations = [(mask_path,   "*.bmp",    args["mask"]),
              (t0_path,     "*.jpg",    args["t0"]),
              (t1_path,     "*.jpg",    args["t1"])]

def rotate_dir(path, extension):
    
    BACKGROUND_COLOR = (0, 0, 0)
    
    if extension == '*.bmp':
        BACKGROUND_COLOR = (255, 255, 255)
    
    for filepath in glob.glob(pjoin(path, extension)):
        image = cv2.imread(filepath)
        h, w, _ = image.shape
        crop_width = int(w / NUM_CROPS)
        center = (crop_width // 2, h // 2)
        for angle in np.arange(0, 360, DEGREE_INCREMENT):
            
            for idx in range(NUM_CROPS):
                
                crop_low = idx * crop_width
                crop_high = idx * crop_width + crop_width
                
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image[:, crop_low : crop_high, :], M, (crop_width, crop_width), borderMode=cv2.BORDER_CONSTANT, borderValue=BACKGROUND_COLOR)
                filename = filepath.split('/')[-1]                              # xxxx.yyy
                filename = filename.split('.')[0]                               # xxxx
                extension_letters = extension.split('.')[-1]                    # yyy
                filename += "_{}_{}.{}".format(idx, angle, extension_letters)   # xxxx_angle.yyy
                output_path = path.replace("PCD", "rotated_PCD")
                os.makedirs(output_path, exist_ok=True)
                cv2.imwrite(pjoin(output_path, filename), rotated)
            

for (path, extension, arg) in operations:
    if arg:
        print("Performing rotation augmentation in path: {}".format(path))
        rotate_dir(path, extension)