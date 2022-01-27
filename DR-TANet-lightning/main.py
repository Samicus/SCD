from params import  encoder_arch, local_kernel_size, stride, \
                    padding, groups, drtam, refinement, \
                    MAX_EPOCHS, CHECKPOINT_DIR, CONFIG
from TANet import TANet
from DataModules import PCDdataModule, OtherDataModule
from pytorch_lightning import Trainer
from os.path import join as pjoin
from aim.pytorch_lightning import AimLogger
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--aim", action="store_true")
parser.add_argument("--cpu", action="store_true")
parsed_args = parser.parse_args()

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")

NUM_SETS = 1
if CONFIG == 'PCD':
    NUM_SETS = 2

NUM_GPU = 1
if parsed_args.cpu:
    NUM_GPU = 0
    
for set_nr in range(NUM_SETS):
    if parsed_args.aim:
        print("Logging data to AIM")
        aim_logger = AimLogger(
        experiment='resnet18_PCD_set{}_{}'.format(set_nr, date_time),
        train_metric_prefix='train_',
        val_metric_prefix='val_',
        test_metric_prefix='test_'
        )
    else:
        aim_logger = None
        
    trainer = Trainer(gpus=NUM_GPU, log_every_n_steps=25, max_epochs=MAX_EPOCHS, 
                      default_root_dir=pjoin(CHECKPOINT_DIR,"set{}".format(set_nr)), 
                      logger=aim_logger)

    if CONFIG == 'PCD':
        data_module = PCDdataModule(set_nr)
    else:
        data_module = OtherDataModule()
    
    len_train_loader = len(data_module.train_dataloader())
    model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, len_train_loader=len_train_loader)
    trainer.fit(model, data_module)
    trainer.test(ckpt_path="best")