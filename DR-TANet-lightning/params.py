config = 2
  
if config == 1:
    MAX_EPOCHS = 20
    NUM_WORKERS = 2
    DATA_DIR = '/home/elias/sam_dev/SCD/TSUNAMI/'
    CHECKPOINT_DIR = '/home/elias/sam_dev/SCD/Checkpoints/tsunami'
    BATCH_SIZE = 16
    SET_NUMBER = 0
    encoder_arch = 'resnet18'
    local_kernel_size = 1
    stride = 1
    padding = 0
    groups = 4
    drtam = False
    refinement = False

if config == 2:
    MAX_EPOCHS = 20
    NUM_WORKERS = 2
    DATA_DIR = '/home/elias/sam_dev/SCD/vl_cmu_cd_binary_mask/'
    CHECKPOINT_DIR = '/home/elias/sam_dev/SCD/Checkpoints/vl_cmu_cd'
    BATCH_SIZE = 16
    SET_NUMBER = 0
    encoder_arch = 'resnet18'
    local_kernel_size = 1
    stride = 1
    padding = 0
    groups = 4
    drtam = True
    refinement = True
          