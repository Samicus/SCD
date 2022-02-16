import os
from os.path import join as pjoin

dirname = os.path.dirname
PCD_DIR = pjoin(dirname(dirname(dirname(__file__))), "PCD/combined")
TSUNAMI_DIR = pjoin(dirname(dirname(dirname(__file__))), "PCD/TSUNAMI")
GSV_DIR = pjoin(dirname(dirname(dirname(__file__))), "PCD/GSV")
CHECKPOINT_DIR = pjoin(dirname(dirname(dirname(__file__))), "Checkpoints")
dir_img = pjoin(dirname(dirname(dirname(__file__))), "dir_img")


MAX_EPOCHS = 150
NUM_WORKERS = 8
BATCH_SIZE = 5
NUM_SETS = 1
encoder_arch = 'resnet18'
local_kernel_size = 1
stride = 1
padding = 0
groups = 4
drtam = True
refinement = True
store_imgs = True