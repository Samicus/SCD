from network.TANet import TANet
from pytorch_lightning import Trainer
from data.DataModules import PCDdataModule
import argparse
import glob
from os.path import join as pjoin
from util import load_config

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--checkpoints", required=True,
	help="path to checkpoints")
parser.add_argument("-e", "--experiment", required=True,
	help="name of experiment")
parser.add_argument("-c", "--configs", required=True,
	help="path to configs")
parser.add_argument("-d", "--dataset", required=True,
	help="TSUNAMI or GSV?")
parser.add_argument("-n", "--set", required=True,
	help="set number")
parsed_args = parser.parse_args()

SET_NUM = int(parsed_args.set)

yaml_path = pjoin(parsed_args.configs, "**/{}.yaml".format(parsed_args.experiment))
config_file = glob.glob(yaml_path)[0]

config = load_config(config_file)
AUG_PARAMS = config["AUGMENTATIONS"]
AUGMENT_ON = config["MISC"]["AUGMENT_ON"]
PCD_CONFIG = config["MISC"]["PCD_CONFIG"]
PRE_PROCESS = config["MISC"]["PRE_PROCESS"]
NUM_WORKERS = config["MISC"]["NUM_WORKERS"]
BATCH_SIZE = config["MISC"]["BATCH_SIZE"]

current_experiment = "{}_PCD_set{}".format(parsed_args.experiment, SET_NUM)
checkpoint = glob.glob(pjoin(parsed_args.checkpoints, current_experiment, "**/**/epoch=*"))[0]
model = TANet.load_from_checkpoint(
                        checkpoint_path=checkpoint,
                        map_location=None,
                        )
trainer = Trainer(gpus=1)
data_module = PCDdataModule(SET_NUM, AUG_PARAMS, AUGMENT_ON, PRE_PROCESS, PCD_CONFIG, NUM_WORKERS, BATCH_SIZE, EVAL=parsed_args.dataset)
trainer.test(model, data_module)