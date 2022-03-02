import os
from os.path import join as pjoin

dirname = os.path.dirname
PCD_DIR = pjoin(dirname(dirname(dirname(__file__))), "PCD")
ROT_PCD_DIR = pjoin(dirname(dirname(dirname(__file__))), "rotated_PCD")
CHECKPOINT_DIR = pjoin(dirname(dirname(dirname(__file__))), "Checkpoints")
dir_img = pjoin(dirname(dirname(dirname(__file__))), "dir_img")

TSUNAMI_DIR = pjoin(PCD_DIR, "TSUNAMI")
GSV_DIR = pjoin(PCD_DIR, "GSV")
ROT_TSUNAMI_DIR = pjoin(ROT_PCD_DIR, "TSUNAMI")
ROT_GSV_DIR = pjoin(ROT_PCD_DIR, "GSV")


MAX_EPOCHS = 100
NUM_WORKERS = 12
BATCH_SIZE = 8
NUM_SETS = 1
encoder_arch = 'resnet18'
local_kernel_size = 3
stride = 1
padding = 1
groups = 4
drtam = False
refinement = False
store_imgs = False

augment_on = False
# MOSAIC Augmentation
mosaic_aug = False
mosaic_th = 0.5
translate = 0.2
scale = [0.1,0.6]

# Random Erase Augmentation
random_erase_aug = False
random_erase_th = 1.0

# Albumentations
albumentations_config = 0

# Rotation
rotation = False
