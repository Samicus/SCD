from network.TANet import TANet
from data.DataModules import VL_CMU_CD_DataModule
from pytorch_lightning import Trainer
from os.path import join as pjoin
from aim.pytorch_lightning import AimLogger
import argparse
import torch
import os
from util import load_config
from pytorch_lightning.callbacks import ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import Trial
import optuna
from optuna.pruners import BasePruner

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
    
# Run-specific augmentation parameters
config_path = parsed_args.config
augmentations = load_config(config_path)["RUN"]
AUGMENT_ON = augmentations["AUGMENT_ON"]
LOG_NAME = config_path.split('.')[0].split('/')[-1]

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
    
def objective(trial: Trial):
    
    checkpoint_callback = ModelCheckpoint(
        os.path.join(CHECKPOINT_DIR, "trial_{}".format(trial.number)), monitor="f1-score"
    )
    
    aim_logger = AimLogger(
        experiment="trial_{}".format(trial.number),
        train_metric_prefix='train_',
        val_metric_prefix='val_',
        test_metric_prefix='test_'
        )
    
    trainer = Trainer(
        logger=aim_logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=MAX_EPOCHS,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="f1-score")],
        #fast_dev_run=True   # DEBUG
    )
    
    data_module = VL_CMU_CD_DataModule(0, augmentations, AUGMENT_ON, NUM_WORKERS, BATCH_SIZE, trial)
    DATASET = "VL_CMU_CD"
    WEIGHT = torch.tensor(4)
    
    EXPERIMENT_NAME = '{}_{}_trial_{}'.format(LOG_NAME, DATASET, trial.number)
    
    model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, EXPERIMENT_NAME, WEIGHT, DETERMINISTIC=DETERMINISTIC)
    trainer.fit(model, data_module)
    
    return trainer.logged_metrics["f1-score"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=1200)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))