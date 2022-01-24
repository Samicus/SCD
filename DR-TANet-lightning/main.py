from TANet import TANet
from DataModules import PcdDataModule, VLCmuCdDataModule
from pytorch_lightning import Trainer
from params import encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, MAX_EPOCHS, CHECKPOINT_DIR, config
from os.path import join as pjoin

#trainer = Trainer(gpus=1, fast_dev_run=True) # Debug

if config == 1:
    for set_nr in range(2):
        trainer = Trainer(gpus=1, max_epochs=MAX_EPOCHS, default_root_dir=pjoin(CHECKPOINT_DIR,"set{}".format(set_nr)))
        data_module = PcdDataModule(set_nr)
        len_train_loader = len(data_module.train_dataloader())
        model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, len_train_loader=len_train_loader)
        trainer.fit(model, data_module)

if config == 2:
    trainer = Trainer(gpus=1, max_epochs=MAX_EPOCHS, default_root_dir=CHECKPOINT_DIR)
    data_module = VLCmuCdDataModule()
    len_train_loader = len(data_module.train_dataloader())
    model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, len_train_loader=len_train_loader)
    trainer.fit(model, data_module) 