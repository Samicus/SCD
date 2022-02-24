from params import  encoder_arch, local_kernel_size, stride, \
                    padding, groups, drtam, refinement, \
                    MAX_EPOCHS, CHECKPOINT_DIR, NUM_SETS
from network.TANet import TANet
from data.DataModules import PCDdataModule
from pytorch_lightning import Trainer, seed_everything
from os.path import join as pjoin
from aim.pytorch_lightning import AimLogger
import argparse
from datetime import datetime

def main(set_nr, data_module, aim_logger):

    trainer = Trainer(gpus=NUM_GPU, log_every_n_steps=5, max_epochs=MAX_EPOCHS, 
                    default_root_dir=pjoin(CHECKPOINT_DIR,"set{}".format(set_nr)),
                    logger=aim_logger, strategy='dp'
                    )
    
    len_train_loader = len(data_module.train_dataloader())
    model = TANet(encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, len_train_loader=len_train_loader)
    trainer.fit(model, data_module)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--aim", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true")
    parsed_args = parser.parse_args()

    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")

    NUM_GPU = 1
    if parsed_args.cpu:
        NUM_GPU = 0
    if parsed_args.multi_gpu:
        NUM_GPU = 2

    for set_nr in range(NUM_SETS):

        data_module = PCDdataModule(set_nr)

        if parsed_args.aim:
            print("Logging data to AIM")
            aim_logger = AimLogger(
            #experiment='{}_PCD_set{}_{}'.format(encoder_arch, set_nr, date_time),
            train_metric_prefix='train_',
            experiment="train_{}_eval_{}".format(data_module.TRAIN_DATASET_NAME, data_module.VAL_DATASET_NAME),
            val_metric_prefix='val_',
            test_metric_prefix='test_'
            )
        else:
            aim_logger = None

        main(set_nr, data_module, aim_logger)