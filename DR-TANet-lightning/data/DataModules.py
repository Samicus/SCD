from pytorch_lightning import LightningDataModule
from os.path import join as pjoin
from torch.utils.data import DataLoader, ConcatDataset
from data.datasets import PCD
import os

dirname = os.path.dirname
PCD_DIR = pjoin(dirname(dirname(dirname(dirname(__file__)))), "PCD")
ROT_PCD_DIR = pjoin(dirname(dirname(dirname(dirname(__file__)))), "rotated_PCD")

TSUNAMI_DIR = pjoin(PCD_DIR, "TSUNAMI")
GSV_DIR = pjoin(PCD_DIR, "GSV")
ROT_TSUNAMI_DIR = pjoin(ROT_PCD_DIR, "TSUNAMI")
ROT_GSV_DIR = pjoin(ROT_PCD_DIR, "GSV")

class PCDdataModule(LightningDataModule):
    def __init__(self, set_nr, aug_params, AUGMENT_ON, PRE_PROCESS, PCD_CONFIG, NUM_WORKERS, BATCH_SIZE):
        self.set_nr = set_nr
        self.aug_params = aug_params
        self.NUM_WORKERS = NUM_WORKERS
        self.BATCH_SIZE = BATCH_SIZE
        
        pre_process = {
            "default": {"TSUNAMI": TSUNAMI_DIR, "GSV": GSV_DIR},
            "paper": {"TSUNAMI": ROT_TSUNAMI_DIR, "GSV": ROT_GSV_DIR}
                       }[PRE_PROCESS]
        
        TSUNAMI = PCD(pjoin(pre_process["TSUNAMI"], "set{}".format(self.set_nr), "train"), self.aug_params, AUGMENT_ON, PCD_CONFIG)
        GSV = PCD(pjoin(pre_process["GSV"], "set{}".format(self.set_nr), "train"), self.aug_params, AUGMENT_ON, PCD_CONFIG)
        self.concat_data = ConcatDataset([TSUNAMI, GSV])
        
        TSUNAMI_val = PCD(pjoin(TSUNAMI_DIR, "set{}".format(self.set_nr), "test"), AUG_PARAMS=self.aug_params, AUGMENT_ON=False, PCD_CONFIG="full")
        GSV_val = PCD(pjoin(GSV_DIR, "set{}".format(self.set_nr), "test"), AUG_PARAMS=self.aug_params, AUGMENT_ON=False, PCD_CONFIG="full")
        self.concat_data_val = ConcatDataset([TSUNAMI_val, GSV_val])
        
    def train_dataloader(self):
        return  DataLoader(self.concat_data,
                           num_workers=self.NUM_WORKERS, 
                           batch_size=self.BATCH_SIZE,
                           shuffle=True)
      
    def test_dataloader(self):
        return DataLoader(PCD(pjoin(TSUNAMI_DIR, "set{}".format(self.set_nr), "test"), AUG_PARAMS=self.aug_params, AUGMENT_ON=False, PCD_CONFIG="full"),
                                          num_workers=self.NUM_WORKERS, batch_size=self.BATCH_SIZE,
                                          shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.concat_data_val,
                          num_workers=self.NUM_WORKERS,
                          batch_size=self.BATCH_SIZE,
                          shuffle=False)
