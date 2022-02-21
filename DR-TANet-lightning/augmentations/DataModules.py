from params import NUM_WORKERS, TSUNAMI_DIR, GSV_DIR, optimized_GSV_DIR, BATCH_SIZE
from pytorch_lightning import LightningDataModule
from os.path import join as pjoin
from torch.utils.data import DataLoader, ConcatDataset
from augmentations import datasets

class PCDdataModule(LightningDataModule):
    def __init__(self, set_nr):
        self.set_nr = set_nr
        
        # IMPORTANT FOR LOG NAME
        self.TRAIN_DATASET_NAME = "optimized_GSV"
        self.VAL_DATASET_NAME = "TSUNAMI"
        
        self.TSUNAMI_dataset = datasets.PCD(pjoin(TSUNAMI_DIR, "set{}".format(self.set_nr), "train"))
        self.GSV_dataset = datasets.PCD(pjoin(GSV_DIR, "set{}".format(self.set_nr), "train"))
        self.optimized_GSV_dataset = datasets.PCD(pjoin(optimized_GSV_DIR, "set{}".format(self.set_nr), "train"))
        self.concatenated_datasets = ConcatDataset([self.TSUNAMI_dataset, self.GSV_dataset])
        #self.combined_dataset = datasets.PCD(pjoin(PCD_DIR, "set{}".format(self.set_nr), "train"))


    def train_dataloader(self):
        return  DataLoader(self.optimized_GSV_dataset,
                           num_workers=NUM_WORKERS, 
                           batch_size=BATCH_SIZE,
                           shuffle=False)
      
    def test_dataloader(self):
        return DataLoader(datasets.PCDeval(pjoin(TSUNAMI_DIR, "set{}".format(self.set_nr), "test")),
                                          num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
                                          shuffle=False)

    def val_dataloader(self):
        return DataLoader(datasets.PCDeval(pjoin(TSUNAMI_DIR, "set{}".format(self.set_nr), "test")),
                                          num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
                                          shuffle=False)
