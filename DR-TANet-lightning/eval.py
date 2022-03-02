from network.TANet import TANet
from pytorch_lightning import Trainer
from data.DataModules import PCDdataModule
import argparse

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