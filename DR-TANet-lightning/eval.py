from TANet import TANet
from pytorch_lightning import Trainer
from DataModules import PCDdataModule
import argparse

""" PATHS

/home/arwin/Downloads/epoch=18931-step=246115.ckpt
/home/arwin/Documents/git/SCD/.aim/resnet18_PCD_set0_02_10_2022_09_10_12/e1356f0a89894c2eaa5cafb7/checkpoints/epoch=199-step=2399.ckpt

"""

parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="ckpt", required=True)
args = parser.parse_args()

for set_nr in range(1):
    model = TANet.load_from_checkpoint(
                            checkpoint_path=args.ckpt,
                            map_location=None,
                            )
    trainer = Trainer(gpus=1)
    data_module = PCDdataModule(set_nr)
    trainer.test(model, data_module)