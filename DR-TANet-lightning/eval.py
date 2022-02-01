from TANet import TANet
from pytorch_lightning import Trainer
from DataModules import PCDdataModule
from params import CHECKPOINT_MODEL_DIR

CHECKPOINT_MODEL_DIR = '/home/arwin/Documents/git/checkpoint_dir/set0/checkpoints/epoch=99-step=2499.ckpt'
HPARAMS_DIR = '/home/arwin/Documents/git/SCD/lightning_logs/version_0/hparams.yaml'


for set_nr in range(1):
    model = TANet.load_from_checkpoint(
                            checkpoint_path=CHECKPOINT_MODEL_DIR,
                            map_location=None,
                            )
    trainer = Trainer(gpus=1)
    data_module = PCDdataModule(set_nr)
    trainer.test(model, data_module)