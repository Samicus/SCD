from pytorch_lightning import LightningDataModule
from os.path import join as pjoin
from torch.utils.data import DataLoader
import datasets

# JSON
NUM_WORKERS = 8
DATA_DIR = '/home/arwin/Downloads/TSUNAMI/'
BATCH_SIZE = 16
SET_NUMBER = 0

class PcdDataModule(LightningDataModule):

    def train_dataloader(self):
        return DataLoader(datasets.pcd(pjoin(DATA_DIR, "set{}".format(SET_NUMBER), "train")),
                                          num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
                                          shuffle=True)

    def test_dataloader(self):
        return DataLoader(datasets.pcd(pjoin(DATA_DIR, "set{}".format(SET_NUMBER), "test")),
                                          num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
                                          shuffle=True)