from TANet import TANet
from dataset_modules.PcdDataModule import PcdDataModule
from pytorch_lightning import Trainer
from params import encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, MAX_EPOCHS, CHECKPOINT_DIR


#trainer = Trainer(gpus=1, fast_dev_run=True) # Debug

for set_nr in range(2):
    trainer = Trainer(gpus=1, max_epochs=MAX_EPOCHS, default_root_dir=pjoin(CHECKPOINT_DIR,"set{}".format(set_nr))
    data_module = PcdDataModule(set_nr)
    len_train_loader = len(data_module.train_dataloader())
    model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, len_train_loader=len_train_loader)
    trainer.fit(model, data_module)

