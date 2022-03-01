from params import NUM_WORKERS, TSUNAMI_DIR, GSV_DIR, ROT_TSUNAMI_DIR, ROT_GSV_DIR, BATCH_SIZE
from pytorch_lightning import LightningDataModule
from os.path import join as pjoin
from torch.utils.data import DataLoader, ConcatDataset
from data import datasets

class PCDdataModule(LightningDataModule):
    def __init__(self, set_nr):
        self.set_nr = set_nr
        
        # IMPORTANT FOR LOG NAME
        self.TRAIN_DATASET_NAME = "rotated_TSUNAMI_and_GSV"
        self.VAL_DATASET_NAME = "TSUNAMI_crop"
        
        self.TSUNAMI_dataset = datasets.PCDcrop(pjoin(TSUNAMI_DIR, "set{}".format(self.set_nr), "train"))
        self.GSV_dataset = datasets.PCDcrop(pjoin(GSV_DIR, "set{}".format(self.set_nr), "train"))
        self.concatenated_datasets = ConcatDataset([self.TSUNAMI_dataset, self.GSV_dataset])

        #self.rotated_TSUNAMI_dataset = datasets.PCDcrop(pjoin(ROT_TSUNAMI_DIR, "set{}".format(self.set_nr), "train"))
        #self.rotated_GSV_dataset = datasets.PCDcrop(pjoin(ROT_GSV_DIR, "set{}".format(self.set_nr), "train"))
        #self.rotated_concatenated_datasets = ConcatDataset([self.rotated_TSUNAMI_dataset, self.rotated_GSV_dataset])


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
