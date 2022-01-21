from parameters import encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, LEARNING_RATE, BATCH_SIZE
from TANet import TANet
from dataset_modules.PCD_data_module import PCD
import pytorch_lightning as pl
from aim.pytorch_lightning import AimLogger
from aim import Run

run_aim = Run()

# Log run parameters
run_aim["hparams"] = {
"learning_rate": LEARNING_RATE,
"batch_size": BATCH_SIZE,
"encoder_arch": encoder_arch,
"local_kernel_size": local_kernel_size,
"stride": stride,
"padding": padding,
"groups": groups,
"drtam": drtam,
"refinement": refinement
}

aim_logger = AimLogger(
    experiment='resnet18_PCD',
    train_metric_prefix='train_',
    val_metric_prefix='val_',
)

#trainer = pl.Trainer(logger=AimLogger(experiment='resnet18_PCD'), gpus=1, fast_dev_run=True) # Debug
trainer = pl.Trainer(gpus=1, logger=aim_logger)
model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement)
data_module = PCD()
trainer.fit(model, data_module)