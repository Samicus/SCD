from params import NUM_WORKERS, TSUNAMI_DIR, GSV_DIR, BATCH_SIZE
from pytorch_lightning import LightningDataModule
from os.path import join as pjoin
from torch.utils.data import DataLoader, ConcatDataset
import datasets

class PCDdataModule(LightningDataModule):
    def __init__(self, set_nr):
        self.set_nr = set_nr
        
        self.TSUNAMI_dataset = datasets.PCD(pjoin(TSUNAMI_DIR, "set{}".format(self.set_nr), "train"))
        self.GSV_dataset = datasets.PCD(pjoin(GSV_DIR, "set{}".format(self.set_nr), "train"))
        self.concatenated_datasets = ConcatDataset([self.TSUNAMI_dataset, self.GSV_dataset])
        #self.combined_dataset = datasets.PCD(pjoin(PCD_DIR, "set{}".format(self.set_nr), "train"))


    def train_dataloader(self):
        return  DataLoader(self.concatenated_datasets,
                           num_workers=NUM_WORKERS, 
                           batch_size=BATCH_SIZE,
                           shuffle=True)
      
    def test_dataloader(self):
        return DataLoader(datasets.PCDeval(pjoin(TSUNAMI_DIR, "set{}".format(self.set_nr), "test")),
                                          num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
                                          shuffle=False)

    def val_dataloader(self):
        return DataLoader(datasets.PCDeval(pjoin(TSUNAMI_DIR, "set{}".format(self.set_nr), "test")),
                                          num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
                                          shuffle=False)
