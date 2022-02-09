from TANet import TANet
from pytorch_lightning import Trainer
from DataModules import PCDdataModule

CHECKPOINT_MODEL_DIR = '/home/arwin/Downloads/epoch=18931-step=246115.ckpt'
#CHECKPOINT_MODEL_DIR = '/home/arwin/Documents/git/checkpoint_dir/DR-TANet_resnet18_ref/pcd/set0/checkpointdir/00048000.pth'
HPARAMS_DIR = '/home/arwin/Documents/git/SCD/lightning_logs/version_0/hparams.yaml'


for set_nr in range(1):
    model = TANet.load_from_checkpoint(
                            checkpoint_path=CHECKPOINT_MODEL_DIR,
                            map_location=None,
                            )
    trainer = Trainer(gpus=1)
    data_module = PCDdataModule(set_nr)
    trainer.test(model, data_module)