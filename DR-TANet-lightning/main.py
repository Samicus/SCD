from network.TANet import TANet
from data.DataModules import PCDdataModule
from pytorch_lightning import Trainer, seed_everything
from os.path import join as pjoin
from aim.pytorch_lightning import AimLogger
import argparse
from datetime import datetime
import torch
import os
from util import load_config

dirname = os.path.dirname
PCD_DIR = pjoin(dirname(dirname(dirname(__file__))), "PCD")
ROT_PCD_DIR = pjoin(dirname(dirname(dirname(__file__))), "rotated_PCD")
CHECKPOINT_DIR = pjoin(dirname(dirname(dirname(__file__))), "Checkpoints")


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", required=True,
	help="path to YAML")
parser.add_argument("--aim", action="store_true")
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--det", action="store_true")
parsed_args = parser.parse_args()

NUM_GPU = 1
if parsed_args.cpu:
    NUM_GPU = 0
    
DETERMINISTIC = False
if parsed_args.det:
    DETERMINISTIC = True
    torch.use_deterministic_algorithms(True)
    
config_path = parsed_args.config
config = load_config(config_path)
hparams = config["HPARAMS"]
misc = config["MISC"]

# Miscellaneous
NUM_SETS = misc["NUM_SETS"]
MAX_EPOCHS = misc["MAX_EPOCHS"]
LOG_NAME = misc["LOG_NAME"]
NUM_WORKERS = misc["NUM_WORKERS"]
BATCH_SIZE = misc["BATCH_SIZE"]
PRE_PROCESS = misc["PRE_PROCESS"]
PCD_CONFIG = misc["PCD_CONFIG"]
AUGMENT_ON = misc["AUGMENT_ON"]

# Hyper Parameters
encoder_arch = hparams["encoder_arch"]
local_kernel_size = hparams["local_kernel_size"]
stride = hparams["stride"]
padding = hparams["padding"]
groups = hparams["groups"]
drtam = hparams["drtam"]
refinement = hparams["refinement"]

# Augmentation parameters
aug_params = config["AUGMENTATIONS"]

for set_nr in range(NUM_SETS):
    
    data_module = PCDdataModule(set_nr, aug_params, AUGMENT_ON, PRE_PROCESS, PCD_CONFIG, NUM_WORKERS, BATCH_SIZE)
    
    if parsed_args.aim:
        print("Logging data to AIM")
        aim_logger = AimLogger(
        experiment='{}_PCD_set{}'.format(LOG_NAME, set_nr),
        train_metric_prefix='train_',
        val_metric_prefix='val_',
        test_metric_prefix='test_'
        )
    else:
        aim_logger = None
        
   
    trainer = Trainer(gpus=NUM_GPU, log_every_n_steps=5, max_epochs=MAX_EPOCHS, 
                      default_root_dir=pjoin(CHECKPOINT_DIR,"set{}".format(set_nr)),
                      logger=aim_logger, deterministic=DETERMINISTIC
                      )
    
    model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, DETERMINISTIC=DETERMINISTIC)
    trainer.fit(model, data_module)