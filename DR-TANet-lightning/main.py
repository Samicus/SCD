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
parser.add_argument("--VL_CMU_CD", action="store_true")
parsed_args = parser.parse_args()

NUM_GPU = 1
if parsed_args.cpu:
    NUM_GPU = 0
    
DETERMINISTIC = False
if parsed_args.det:
    DETERMINISTIC = True
    torch.use_deterministic_algorithms(True)
    
# Run-specific augmentation parameters
config_path = parsed_args.config
augmentations = load_config(config_path)["RUN"]
AUGMENT_ON = augmentations["AUGMENT_ON"]
LOG_NAME = config_path.split('.')[0]

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


for set_nr in range(0, NUM_SETS):
    
    if parsed_args.VL_CMU_CD:
        data_module = VL_CMU_CD_DataModule(set_nr, augmentations, AUGMENT_ON, NUM_WORKERS, BATCH_SIZE)
        DATASET = "VL_CMU_CD"
    else:
        data_module = PCDdataModule(set_nr, augmentations, AUGMENT_ON, PRE_PROCESS, PCD_CONFIG, NUM_WORKERS, BATCH_SIZE)
        DATASET = "PCD"

    EXPERIMENT_NAME = '{}_{}_set{}'.format(LOG_NAME, DATASET, set_nr)
    
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

    early_stop_callback = EarlyStopping(monitor="f1-score", min_delta=0.00, patience=50, verbose=False, mode="max")
    checkpoint_callback = ModelCheckpoint(
        monitor="f1-score",
        #dirpath="my/path/",
        #filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        save_last=True,
        mode="max",
    )
    trainer = Trainer(gpus=NUM_GPU, log_every_n_steps=5, max_epochs=MAX_EPOCHS, 
                      default_root_dir=pjoin(CHECKPOINT_DIR,"set{}".format(set_nr)),
                      logger=aim_logger, deterministic=DETERMINISTIC, callbacks=[early_stop_callback, checkpoint_callback],
                      )
    
    model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, EXPERIMENT_NAME, DETERMINISTIC=DETERMINISTIC)
    trainer.fit(model, data_module)