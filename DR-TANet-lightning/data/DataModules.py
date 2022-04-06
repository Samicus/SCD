from pytorch_lightning import LightningDataModule
from os.path import join as pjoin
from torch.utils.data import DataLoader, ConcatDataset
from data.datasets import VL_CMU_CD, PCD
import os

dirname = os.path.dirname
PCD_DIR = pjoin(dirname(dirname(dirname(dirname(__file__)))), "PCD")
ROT_PCD_DIR = pjoin(dirname(dirname(dirname(dirname(__file__)))), "rotated_PCD")
VL_CMU_CD_DIR = pjoin(dirname(dirname(dirname(dirname(__file__)))), "vl_cmu_cd_binary_mask")
VL_CMU_CD_SMALL_DIR = pjoin(dirname(dirname(dirname(dirname(__file__)))), "vl_cmu_cd_binary_mask_small")

TSUNAMI_DIR = pjoin(PCD_DIR, "TSUNAMI")
GSV_DIR = pjoin(PCD_DIR, "GSV")
ROT_TSUNAMI_DIR = pjoin(ROT_PCD_DIR, "TSUNAMI")
ROT_GSV_DIR = pjoin(ROT_PCD_DIR, "GSV")

class PCDdataModule(LightningDataModule):
    def __init__(self, set_nr, augmentations, AUGMENT_ON, PRE_PROCESS, PCD_CONFIG, NUM_WORKERS, BATCH_SIZE, trial=None, EVAL="PCD"):
        self.set_nr = set_nr
        self.augmentations = augmentations
        self.NUM_WORKERS = NUM_WORKERS
        self.BATCH_SIZE = BATCH_SIZE
        
        pre_process = {
            "default": {"TSUNAMI": TSUNAMI_DIR, "GSV": GSV_DIR},
            "paper": {"TSUNAMI": ROT_TSUNAMI_DIR, "GSV": ROT_GSV_DIR}
                       }[PRE_PROCESS]
        
        TSUNAMI = PCD(pjoin(pre_process["TSUNAMI"], "set{}".format(self.set_nr), "train"), self.augmentations, AUGMENT_ON, PCD_CONFIG, trial=trial)
        GSV = PCD(pjoin(pre_process["GSV"], "set{}".format(self.set_nr), "train"), self.augmentations, AUGMENT_ON, PCD_CONFIG, trial=trial)
        self.concat_data = ConcatDataset([TSUNAMI, GSV])
        
        TSUNAMI_test = PCD(pjoin(TSUNAMI_DIR, "set{}".format(self.set_nr), "test"), augmentations=self.augmentations, AUGMENT_ON=False, PCD_CONFIG="full")
        GSV_test = PCD(pjoin(GSV_DIR, "set{}".format(self.set_nr), "test"), augmentations=self.augmentations, AUGMENT_ON=False, PCD_CONFIG="full")
        concat_data_val = ConcatDataset([TSUNAMI_test, GSV_test])
        
        if EVAL == 'PCD':
            self.test_data = concat_data_val
        elif EVAL == 'TSUNAMI':
            self.test_data = TSUNAMI_test 
        elif EVAL == 'GSV':
            self.test_data = GSV_test
        else:
            print("Choose a valid dataset for eval")
            exit(1)
            
    def train_dataloader(self):
        return  DataLoader(self.concat_data,
                           num_workers=self.NUM_WORKERS, 
                           batch_size=self.BATCH_SIZE,
                           shuffle=True)
      
    def test_dataloader(self):
        return DataLoader(self.test_data,
                          num_workers=self.NUM_WORKERS,
                          batch_size=self.BATCH_SIZE,
                          shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.test_data,
                          num_workers=self.NUM_WORKERS,
                          batch_size=self.BATCH_SIZE,
                          shuffle=False)

class VL_CMU_CD_DataModule(LightningDataModule):
    def __init__(self, set_nr, augmentations, AUGMENT_ON, NUM_WORKERS, BATCH_SIZE, trial):
        self.set_nr = set_nr
        self.augmentations = augmentations
        self.NUM_WORKERS = NUM_WORKERS
        self.BATCH_SIZE = BATCH_SIZE
        
        self.VL_CMU_CD = VL_CMU_CD(pjoin(VL_CMU_CD_SMALL_DIR, "set{}".format(self.set_nr), "train"), self.augmentations, AUGMENT_ON, trial=trial)
        self.VL_CMU_CD_test = VL_CMU_CD(pjoin(VL_CMU_CD_SMALL_DIR, "set{}".format(self.set_nr), "test"), self.augmentations, AUGMENT_ON=False)
        self.VL_CMU_CD_val = VL_CMU_CD(pjoin(VL_CMU_CD_SMALL_DIR, "set{}".format(self.set_nr), "test"), self.augmentations, AUGMENT_ON=False)
        
    def train_dataloader(self):
        return  DataLoader(self.VL_CMU_CD,
                           num_workers=self.NUM_WORKERS, 
                           batch_size=self.BATCH_SIZE,
                           shuffle=True)
      
    def test_dataloader(self):
        return DataLoader(self.VL_CMU_CD_test,
                          num_workers=self.NUM_WORKERS, 
                          batch_size=self.BATCH_SIZE,
                          shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.VL_CMU_CD_val,
                          num_workers=self.NUM_WORKERS,
                          batch_size=self.BATCH_SIZE,
                          shuffle=False)