from network.TANet import TANet
from data.datasets import PCD
import argparse
from util import load_config
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--data_path", required=True,
	help="path to checkpoint")
parser.add_argument("-i", "--checkpoint", required=True,
	help="path to checkpoint")
parser.add_argument("--cpu", action="store_true")
parser.add_argument("-c", "--config", required=True,
	help="path to config")
parsed_args = parser.parse_args()

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

pcd_data = PCD(parsed_args.data_path, augmentations, AUGMENT_ON, PCD_CONFIG)
input_data, _ = pcd_data.get_random_image()
input_data = torch.tensor(input_data).unsqueeze(dim=0)

traced_script_module = torch.jit.trace(model, input_data)

traced_script_module.save("mobile_application/traced_DR_TANet_ref_1024x224.pt")