from TANet import TANet
from dataset_modules.PcdDataModule import PcdDataModule
from pytorch_lightning import Trainer

# JSON
encoder_arch = 'resnet18'
local_kernel_size = 1
stride = 1
padding = 0
groups = 4
drtam = False
refinement = False

trainer = Trainer(gpus=1, fast_dev_run=True) # Debug
#trainer = Trainer(gpus=1)
model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement)
data_module = PcdDataModule()
trainer.fit(model, data_module)