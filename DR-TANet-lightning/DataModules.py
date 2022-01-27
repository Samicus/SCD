from pytorch_lightning import LightningDataModule
from os.path import join as pjoin
from torch.utils.data import DataLoader
import datasets
from params import NUM_WORKERS, DATA_DIR, BATCH_SIZE, SET_NUMBER


class PCD(LightningDataModule):
    def __init__(self, set_nr):
        self.set_nr = set_nr


    def train_dataloader(self):
        return  DataLoader(datasets.pcd(pjoin(DATA_DIR, "set{}".format(self.set_nr), "train")),
                                          num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
                                          shuffle=True)
      
    def test_dataloader(self):
        return DataLoader(datasets.pcd(pjoin(DATA_DIR, "set{}".format(self.set_nr), "test")),
                                          num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
                                          shuffle=False)

    def val_dataloader(self):
        return DataLoader(datasets.pcd(pjoin(DATA_DIR, "set{}".format(self.set_nr), "test")),
                                          num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
                                          shuffle=False)
    
class OtherDataModule(LightningDataModule):
    
    
    def train_dataloader(self):
        return  DataLoader(datasets.vl_cmu_cd(pjoin(DATA_DIR, "train")),
                                          num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
                                          shuffle=True)
      

    def val_dataloader(self):
        return DataLoader(datasets.vl_cmu_cd_eval(pjoin(DATA_DIR, 'test')),
                                          num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
                                          shuffle=True)