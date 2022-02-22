import os
from os.path import join as pjoin

dirname = os.path.dirname
#PCD_DIR = pjoin(dirname(dirname(dirname(__file__))), "PCD/combined")
TSUNAMI_DIR = pjoin(dirname(dirname(dirname(__file__))), "PCD/TSUNAMI")
GSV_DIR = pjoin(dirname(dirname(dirname(__file__))), "PCD/GSV")
optimized_GSV_DIR = pjoin(dirname(dirname(dirname(__file__))), "PCD/optimized_GSV")
CHECKPOINT_DIR = pjoin(dirname(dirname(dirname(__file__))), "Checkpoints")
dir_img = pjoin(dirname(dirname(dirname(__file__))), "dir_img")


MAX_EPOCHS = 200
NUM_WORKERS = 8
BATCH_SIZE = 8
NUM_SETS = 1
encoder_arch = 'resnet18'
local_kernel_size = 7
stride = 1
padding = 3
groups = 4
drtam = True
refinement = True
store_imgs = True

degrees = 0.373
translate = 0.245
scale = 0.898
shear = 0.602
perspective = 0.0

# Augments
mosaic_aug = False

