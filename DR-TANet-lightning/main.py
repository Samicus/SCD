from parameters import encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement
from TANet import TANet
from dataset_modules.PCD_data_module import PCD
import pytorch_lightning as pl
from aim.pytorch_lightning import AimLogger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--aim", action="store_true")
parsed_args = parser.parse_args()

aim_logger = AimLogger(
    experiment='resnet18_PCD',
    train_metric_prefix='train_',
    val_metric_prefix='val_',
    test_metric_prefix='test_'
)

model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement)
data_module = PCD()

if parsed_args.aim:
    print("Logging data to AIM")
    trainer = pl.Trainer(gpus=1, logger=aim_logger)
else:
    trainer = pl.Trainer(gpus=1)

trainer.fit(model, data_module)