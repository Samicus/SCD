CONFIG = 'PCD'

if CONFIG == 'PCD':
    MAX_EPOCHS = 100
    NUM_WORKERS = 8
    DATA_DIR = '/home/arwin/Downloads/TSUNAMI/'
    CHECKPOINT_DIR = '/home/arwin/Documents/git/checkpoint_dir/'
    CHECKPOINT_MODEL_DIR = '/home/arwin/Documents/git/checkpoint_dir/set0/checkpoints/epoch=99-step=2499.ckpt'
    HPARAMS_DIR = '/home/arwin/Documents/git/SCD/lightning_logs/version_0/hparams.yaml'
    BATCH_SIZE = 4
    SET_NUMBER = 0
    encoder_arch = 'resnet18'
    local_kernel_size = 1
    stride = 1
    padding = 0
    groups = 4
    drtam = True
    refinement = True
    store_imgs = True

else:
    MAX_EPOCHS = 20
    NUM_WORKERS = 8
    DATA_DIR = '/home/elias/sam_dev/vl_cmu_cd_binary_mask/'
    #DATA_DIR = 
    CHECKPOINT_DIR = '/home/elias/sam_dev/Checkpoints/vl_cmu_cd'
    #CHECKPOINT_DIR = 
    BATCH_SIZE = 16
    SET_NUMBER = 0
    encoder_arch = 'resnet18'
    local_kernel_size = 1
    stride = 1
    padding = 0
    groups = 4
    drtam = True
    refinement = True
    store_imgs = False