from network.TANet import TANet
from pytorch_lightning import Trainer
from data.DataModules import PCDdataModule
from util import load_config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="ckpt", required=True)
args = parser.parse_args()
parser.add_argument("-c", "--config", required=True,
	help="path to YAML")
parser.add_argument("-d", "--dataset", required=True,
	help="TSUNAMI or GSV?")
parsed_args = parser.parse_args()

config_path = parsed_args.config
config = load_config(config_path)
NUM_SETS = config["MISC"]["NUM_SETS"]

for set_nr in range(NUM_SETS):
    model = TANet.load_from_checkpoint(
                            checkpoint_path=args.ckpt,
                            map_location=None,
                            )
    trainer = Trainer(gpus=1)
    data_module = PCDdataModule(set_nr, EVAL=parsed_args.dataset)
    trainer.test(model, data_module)