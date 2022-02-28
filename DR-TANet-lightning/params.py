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


MAX_EPOCHS = 200
NUM_WORKERS = 8
BATCH_SIZE = 1
NUM_SETS = 1
encoder_arch = 'resnet18'
local_kernel_size = 7
stride = 1
padding = 3
groups = 4
drtam = True
refinement = True
store_imgs = True

augment_on = True
# MOSAIC Augmentation
mosaic_aug = False
mosaic_th = 0.5
translate = 0.2
scale = [0.1,0.6]

# Random Erase Augmentation
random_erase_aug = False
random_erase_th = 1.0

# Albumentations
albumentations_config = 1