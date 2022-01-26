from TANet import TANet
from DataModules import PCD, VLCmuCdDataModule
from pytorch_lightning import Trainer
from params import encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, MAX_EPOCHS, CHECKPOINT_DIR, config
from os.path import join as pjoin
from aim.pytorch_lightning import AimLogger
import argparse
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument("--aim", action="store_true")
parsed_args = parser.parse_args()

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")



if config == 1:
    for set_nr in range(2):
        aim_logger = AimLogger(
            experiment='resnet18_PCD_set{}_{}'.format(set_nr, date_time),
            train_metric_prefix='train_',
            val_metric_prefix='val_',
            test_metric_prefix='test_'
        )
        if parsed_args.aim:
            print("Logging data to AIM")
            trainer = Trainer(gpus=1, max_epochs=MAX_EPOCHS, default_root_dir=pjoin(CHECKPOINT_DIR,"set{}".format(set_nr)), logger=aim_logger)
        else:
            trainer = Trainer(gpus=1, max_epochs=MAX_EPOCHS, default_root_dir=pjoin(CHECKPOINT_DIR,"set{}".format(set_nr)))
        data_module = PCD(set_nr)
        len_train_loader = len(data_module.train_dataloader())
        model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, len_train_loader, set_nr)
        trainer.fit(model, data_module)
        trainer.test(ckpt_path="best")
        

if config == 2:
    aim_logger = AimLogger(
            experiment='resnet18_PCD_set_{}'.format(date_time),
            train_metric_prefix='train_',
            val_metric_prefix='val_',
            test_metric_prefix='test_'
        )
    if parsed_args.aim:
        print("Logging data to AIM")
        trainer = Trainer(gpus=1, max_epochs=MAX_EPOCHS, default_root_dir=CHECKPOINT_DIR, logger=aim_logger)
    else:
        trainer = Trainer(gpus=1, max_epochs=MAX_EPOCHS, default_root_dir=CHECKPOINT_DIR)

    data_module = VLCmuCdDataModule()
    len_train_loader = len(data_module.train_dataloader())
    model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, len_train_loader=len_train_loader)
    trainer.fit(model, data_module) 



