from network.TANet import TANet
from data.DataModules import PCDdataModule, VL_CMU_CD_DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from os.path import join as pjoin
from aim.pytorch_lightning import AimLogger
import argparse
import torch
import os
from util import load_config
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", required=True,
	help="path to YAML")
parser.add_argument("--frac", required=False,
	help="fraction of data to train on")
parser.add_argument("--res", required=False,
	help="path to checkpoint for resuming training")
parser.add_argument("--aim", action="store_true")
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--det", action="store_true")
parser.add_argument("--VL_CMU_CD", action="store_true")
parsed_args = parser.parse_args()

NUM_GPU = 1
if parsed_args.cpu:
    NUM_GPU = 0
    
DETERMINISTIC = False
if parsed_args.det:
    DETERMINISTIC = True
    torch.use_deterministic_algorithms(True)
    
PCD_FRACTION = 1.0
if parsed_args.frac:
    PCD_FRACTION = float(parsed_args.frac)

CKPT_PATH = None
if parsed_args.res:
    CKPT_PATH = parsed_args.res

# Run-specific augmentation parameters
config_path = parsed_args.config
augmentations = load_config(config_path)["RUN"]
AUGMENT_ON = augmentations["AUGMENT_ON"]
LOG_NAME = config_path.split('.')[0].split('/')[-1]

# PCD or VL_CMU_CD settings
config_path = "DR-TANet-lightning/config/PCD.yaml"
if parsed_args.VL_CMU_CD:
    config_path = "DR-TANet-lightning/config/VL_CMU_CD.yaml"
config = load_config(config_path)
hparams = config["HPARAMS"]
misc = config["MISC"]

# Miscellaneous
NUM_SETS = misc["NUM_SETS"]
MAX_EPOCHS = misc["MAX_EPOCHS"]
NUM_WORKERS = misc["NUM_WORKERS"]
BATCH_SIZE = misc["BATCH_SIZE"]
PRE_PROCESS = misc["PRE_PROCESS"]
PCD_CONFIG = misc["PCD_CONFIG"]

# Hyper Parameters
encoder_arch = hparams["encoder_arch"]
local_kernel_size = hparams["local_kernel_size"]
stride = hparams["stride"]
padding = hparams["padding"]
groups = hparams["groups"]
drtam = hparams["drtam"]
refinement = hparams["refinement"]

VALIDATION_STEP = 1
#if parsed_args.aim:
#    VALIDATION_STEP = 10

#start_set = 0
#if PCD_FRACTION == 0.75 and "M_CP_RE_A" in LOG_NAME:
#    start_set = 3
#elif PCD_FRACTION == 0.25 and "vanilla" in LOG_NAME:
#    start_set = 1

for set_nr in range(0, 1):
    
    if parsed_args.VL_CMU_CD:
        data_module = VL_CMU_CD_DataModule(set_nr, augmentations, AUGMENT_ON, NUM_WORKERS, BATCH_SIZE)
        DATASET = "VL_CMU_CD"
    else:
        data_module = PCDdataModule(set_nr, augmentations, AUGMENT_ON, PRE_PROCESS, PCD_CONFIG, PCD_FRACTION, NUM_WORKERS, BATCH_SIZE)
        DATASET = "PCD"

    EXPERIMENT_NAME = '{}_{}_{}_frac({})_set{}'.format(LOG_NAME, DATASET, PCD_CONFIG, PCD_FRACTION, set_nr)
    
    if parsed_args.aim:
        print("Logging data to AIM")
        aim_logger = AimLogger(
        experiment=EXPERIMENT_NAME,
        train_metric_prefix='train_',
        val_metric_prefix='val_',
        test_metric_prefix='test_'
        )
    else:
        aim_logger = None

    checkpoint_callback = ModelCheckpoint(
        monitor="f1-score",
        save_top_k=1,
        save_last=True,
        mode="max",
    )
    
    trainer = Trainer(gpus=NUM_GPU, max_epochs=MAX_EPOCHS,
                logger=aim_logger, deterministic=DETERMINISTIC, callbacks=[checkpoint_callback],
                check_val_every_n_epoch=VALIDATION_STEP,
                default_root_dir="checkpoints/set{}".format(set_nr),
                log_every_n_steps=5, min_epochs=MAX_EPOCHS/2,
                resume_from_checkpoint = CKPT_PATH
                )
    
    model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, EXPERIMENT_NAME, DETERMINISTIC=DETERMINISTIC)
    trainer.fit(model, data_module)