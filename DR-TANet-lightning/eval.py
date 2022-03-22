from network.TANet import TANet
from pytorch_lightning import Trainer
from data.DataModules import PCDdataModule
import argparse
import glob
from os.path import join as pjoin
from util import load_config

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--checkpoint", required=True,
	help="path to checkpoints")
parser.add_argument("-c", "--config", required=True,
	help="path to config")
parser.add_argument("-d", "--dataset", required=True,
	help="TSUNAMI or GSV?")
parser.add_argument("-n", "--set", required=True,
	help="set number")
parsed_args = parser.parse_args()

SET_NUM = int(parsed_args.set)

# Run-specific augmentation parameters
yaml_path = parsed_args.config
augmentations = load_config(yaml_path)["RUN"]
AUGMENT_ON = augmentations["AUGMENT_ON"]

config_path = "DR-TANet-lightning/config/PCD.yaml"
config = load_config(config_path)
misc = config["MISC"]

# Miscellaneous
NUM_WORKERS = misc["NUM_WORKERS"]
BATCH_SIZE = misc["BATCH_SIZE"]
PRE_PROCESS = misc["PRE_PROCESS"]
PCD_CONFIG = misc["PCD_CONFIG"]

model = TANet.load_from_checkpoint(
                        checkpoint_path=parsed_args.checkpoint,
                        map_location=None,
                        )
trainer = Trainer(gpus=1)
data_module = PCDdataModule(SET_NUM, augmentations, AUGMENT_ON, PRE_PROCESS, PCD_CONFIG, NUM_WORKERS, BATCH_SIZE, EVAL=parsed_args.dataset)
trainer.test(model, data_module)