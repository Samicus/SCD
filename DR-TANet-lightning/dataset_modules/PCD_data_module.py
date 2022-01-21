from parameters import NUM_WORKERS, DATA_DIR, BATCH_SIZE, SET_NUMBER
from pytorch_lightning import LightningDataModule
from os.path import join as pjoin
from torch.utils.data import DataLoader
import datasets

class PCD(LightningDataModule):

    def train_dataloader(self):
        return DataLoader(datasets.pcd(pjoin(DATA_DIR, "set{}".format(SET_NUMBER), "train")),
                                          num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
                                          shuffle=True)

    def test_dataloader(self):
        return DataLoader(datasets.pcd(pjoin(DATA_DIR, "set{}".format(SET_NUMBER), "test")),
                                          num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
                                          shuffle=True)