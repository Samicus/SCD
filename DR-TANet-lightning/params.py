import os
from os.path import join as pjoin

dirname = os.path.dirname
PCD_DIR = pjoin(dirname(dirname(dirname(__file__))), "PCD")
CHECKPOINT_DIR = pjoin(dirname(dirname(dirname(__file__))), "Checkpoints")
dir_img = pjoin(dirname(dirname(dirname(__file__))), "dir_img")

TSUNAMI_DIR = pjoin(PCD_DIR, "TSUNAMI")
GSV_DIR = pjoin(PCD_DIR, "GSV")
ROT_TSUNAMI_DIR = pjoin(PCD_DIR, "rotated_TSUNAMI")
ROT_GSV_DIR = pjoin(PCD_DIR, "rotated_GSV")


MAX_EPOCHS = 1000
NUM_WORKERS = 12
BATCH_SIZE = 8
NUM_SETS = 1
encoder_arch = 'resnet18'
local_kernel_size = 7
stride = 1
padding = 3
groups = 4
drtam = False
refinement = False
store_imgs = False


# MOSAIC Augmentation
mosaic_aug = True
mosaic_th = 1.0
translate = 0.1
rotation = 180
scale = [0.1, 1]

# Random Erase Augmentation
random_erase_aug = False
random_erase_th = 1.0