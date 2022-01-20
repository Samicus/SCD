from TANet import TANet
from dataset_modules.PcdDataModule import PcdDataModule
from pytorch_lightning import Trainer
from params import encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement


trainer = Trainer(gpus=1, fast_dev_run=True) # Debug
#trainer = Trainer(gpus=1)

data_module = PcdDataModule()
len_train_loader = len(data_module.train_dataloader())
model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, len_train_loader=len_train_loader)
trainer.fit(model, data_module)