import os
from os.path import join as pjoin

CONFIG = 'PCD'

dirname = os.path.dirname
DATA_DIR = pjoin(dirname(dirname(dirname(__file__))), "TSUNAMI")
CHECKPOINT_DIR = pjoin(dirname(dirname(dirname(__file__))), "Checkpoints")
dir_img = pjoin(dirname(dirname(dirname(__file__))), "dir_img")


if CONFIG == 'PCD':
    MAX_EPOCHS = 5000
    NUM_WORKERS = 12
    BATCH_SIZE = 8
    SET_NUMBER = 0
    encoder_arch = 'resnet18'
    local_kernel_size = 7
    stride = 1
    padding = 3
    groups = 4
    drtam = True
    refinement = True
    store_imgs = True

else:
    MAX_EPOCHS = 20
    NUM_WORKERS = 8
    BATCH_SIZE = 16
    SET_NUMBER = 0
    encoder_arch = 'resnet18'
    local_kernel_size = 1
    stride = 1
    padding = 0
    groups = 4
    drtam = True
    refinement = True
    store_imgs = False