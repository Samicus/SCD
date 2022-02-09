import os
from os.path import join as pjoin

dirname = os.path.dirname
DATA_DIR = pjoin(dirname(dirname(dirname(__file__))), "TSUNAMI")
CHECKPOINT_DIR = pjoin(dirname(dirname(dirname(__file__))), "Checkpoints")
dir_img = pjoin(dirname(dirname(dirname(__file__))), "dir_img")


MAX_EPOCHS = 200
NUM_WORKERS = 8
BATCH_SIZE = 8
NUM_SETS = 1
encoder_arch = 'resnet18'
local_kernel_size = 1
stride = 1
padding = 0
groups = 4
drtam = True
refinement = False
store_imgs = True