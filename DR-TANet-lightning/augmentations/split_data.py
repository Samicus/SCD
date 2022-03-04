import argparse
from os import makedirs
from os.path import join as pjoin
from PIL import Image
import glob
from shutil import copy

""" Required dataset structure

dataset
│   ├── mask/       # *.bmp
│   ├── t0/         # *.jpg
|   ├── t1/         # *.jpg

Usage:
    python3 split_pcd.py -i /path/to/dataset

"""

NUM_SETS = 5

# Construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--dataset", required=True,
	help="path to the dataset")
parser.add_argument("-o", "--output", required=True,
	help="path to the desired output path")
args = vars(parser.parse_args())

DATA_PATH = args["dataset"]
OUTPUT_PATH = args["output"]

mask_path = pjoin(DATA_PATH, "mask")
t0_path = pjoin(DATA_PATH, "t0")
t1_path = pjoin(DATA_PATH, "t1")

operations = [(mask_path,   "*.bmp"),
              (t0_path,     "*.jpg"),
              (t1_path,     "*.jpg")]


for set_nr in range(NUM_SETS):

    for (path, extension) in operations:
        
        data = sorted(glob.glob(pjoin(path, extension)))
        
        num_files = len(data)
        split = 0.2 * num_files
        
        test_low = int(set_nr * split)
        test_high = int(test_low + split)
        
        train_path = pjoin(OUTPUT_PATH, "set{}/train".format(set_nr), path.split('/')[-1])
        test_path = pjoin(OUTPUT_PATH, "set{}/test".format(set_nr), path.split('/')[-1])
        
        makedirs(train_path, exist_ok=True)
        makedirs(test_path, exist_ok=True)
        
        for idx, filepath in enumerate(data):
            
            if test_low <= idx < test_high:
                copy(filepath, test_path)
            else:
                copy(filepath, train_path   )