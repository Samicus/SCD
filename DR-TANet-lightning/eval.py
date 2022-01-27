from TANet import TANet
from pytorch_lightning import Trainer
from DataModules import PCDdataModule


for set_nr in range(2):
    model = TANet.load_from_checkpoint(
                            checkpoint_path="/home/elias/sam_dev/Checkpoints/tsunami/set0/lightning_logs/version_2/checkpoints/epoch=99-step=699.ckpt",
                            hparams_file="/home/elias/sam_dev/Checkpoints/tsunami/set0/lightning_logs/version_3/hparams.yaml",
                            map_location=None,
                            )    
    trainer = Trainer()
    trainer.test(model, PCDdataModule(set_nr))



