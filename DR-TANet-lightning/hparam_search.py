from network.TANet import TANet
from data.DataModules import PCDdataModule
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
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice
from matplotlib import pyplot as plt
import logging
import sys
import gc

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

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
        log_every_n_steps=5,
        min_epochs=10
        #fast_dev_run=True   # DEBUG
    )
    
    data_module = PCDdataModule(0, augmentations, AUGMENT_ON, PRE_PROCESS, PCD_CONFIG, NUM_WORKERS, BATCH_SIZE, trial)
    DATASET = "PCD"
    WEIGHT = torch.tensor(2.52)
    
    EXPERIMENT_NAME = '{}_{}_trial_{}'.format(LOG_NAME, DATASET, trial.number)
    
    model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, EXPERIMENT_NAME, WEIGHT, DETERMINISTIC=DETERMINISTIC)
    trainer.fit(model, data_module)
    
    return trainer.logged_metrics["f1-score"]

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "TANet_(3x3)_augmentations" # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize")
study.optimize(objective, n_trials=10, gc_after_trial=True)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

plt.figure()
plot_contour(study)
plt.savefig("Contour.png")
plt.figure()
plot_edf(study)
plt.savefig("EDF.png")
plt.figure()
plot_intermediate_values(study)
plt.savefig("Intermediate.png")
plt.figure()
plot_optimization_history(study)
plt.savefig("Optimization_History.png")
plt.figure()
plot_parallel_coordinate(study)
plt.savefig("Parallell_Coordinate.png")
plt.figure()
plot_param_importances(study)
plt.savefig("Param_Importance.png")
plt.figure()
plot_slice(study)
plt.savefig("Slice.png")
