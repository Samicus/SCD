from network.TANet import TANet
from data.DataModules import PCDdataModule
from pytorch_lightning import Trainer
from os.path import join as pjoin
from aim.pytorch_lightning import AimLogger
import argparse
import torch
import os
from util import load_config, F1tracker
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import Trial
import optuna
from optuna.pruners import PatientPruner, MedianPruner

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

config_path = "DR-TANet-lightning/config/PCD.yaml"
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

    if parsed_args.aim:
    
        aim_logger = AimLogger(
            experiment="trial_{}".format(trial.number),
            train_metric_prefix='train_',
            val_metric_prefix='val_',
            test_metric_prefix='test_'
            )
    else:
        aim_logger = None

    f1_tracker = F1tracker()
    
    trainer = Trainer(
        logger=aim_logger,
        max_epochs=MAX_EPOCHS,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="f1-score"), 
        f1_tracker
        ],
        log_every_n_steps=10
        #fast_dev_run=True   # DEBUG
    )
    
    data_module = PCDdataModule(0, augmentations, AUGMENT_ON, PRE_PROCESS, PCD_CONFIG, NUM_WORKERS, BATCH_SIZE, trial)
    DATASET = "PCD"
    
    EXPERIMENT_NAME = '{}_{}_trial_{}'.format(LOG_NAME, DATASET, trial.number)
    
    model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, EXPERIMENT_NAME, DETERMINISTIC=DETERMINISTIC)
    trainer.fit(model, data_module)

    highest_f1 = max(f1_tracker.f1_scores)
    
    return highest_f1
    #print("\nHighest F1-Score of trial{}: {}\n".format(trial.trial_id, highest_f1))
    #return trainer.logged_metrics["f1-score"]

patient_pruner = PatientPruner(MedianPruner(), patience=20)

study_name = 'params'  # Unique identifier of the study.
study = optuna.create_study(study_name=study_name, 
                            storage='sqlite:///params.db', 
                            direction="maximize",
                            pruner=patient_pruner,
                            load_if_exists=True)
study.optimize(objective, n_trials=1000)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))