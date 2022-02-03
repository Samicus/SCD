from TANet import TANet
from pytorch_lightning import Trainer
from DataModules import PCDdataModule

CHECKPOINT_MODEL_DIR = '/home/arwin/Documents/git/SCD/.aim/resnet18_PCD_set0_02_03_2022_13_04_32/d3b38d77731c4abe8b33d381/checkpoints/epoch=199-step=4999.ckpt'
HPARAMS_DIR = '/home/arwin/Documents/git/SCD/lightning_logs/version_0/hparams.yaml'


for set_nr in range(1):
    model = TANet.load_from_checkpoint(
                            checkpoint_path=CHECKPOINT_MODEL_DIR,
                            map_location=None,
                            )
    trainer = Trainer(gpus=1)
    data_module = PCDdataModule(set_nr)
    trainer.test(model, data_module)