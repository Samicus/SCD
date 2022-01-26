from TANet import TANet
from DataModules import PCD, VLCmuCdDataModule
from pytorch_lightning import Trainer
from params import encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, MAX_EPOCHS, CHECKPOINT_DIR, config
from os.path import join as pjoin
from aim.pytorch_lightning import AimLogger
import argparse
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument("--aim", action="store_true")
parsed_args = parser.parse_args()

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")



for set_nr in range(2):
    model = TANet.load_from_checkpoint(
                            checkpoint_path="/home/elias/sam_dev/Checkpoints/tsunami/set0/lightning_logs/version_2/checkpoints/epoch=99-step=699.ckpt",
                            hparams_file="/home/elias/sam_dev/Checkpoints/tsunami/set0/lightning_logs/version_2/hparams.yaml",
                            map_location=None,
                            )    
    trainer = Trainer()
    trainer.test(model)



