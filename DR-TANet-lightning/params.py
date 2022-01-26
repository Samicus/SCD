from socketserver import DatagramRequestHandler


config = 1
  

if config == 1:
    MAX_EPOCHS = 100
    NUM_WORKERS = 1
    DATA_DIR = '/home/samnehme/Dev/SCD_project/TSUNAMI'

    CHECKPOINT_DIR = '/home/samnehme/Dev/SCD_project/Checkpoints'
    BATCH_SIZE = 1
    SET_NUMBER = 0
    encoder_arch = 'resnet18'
    local_kernel_size = 1
    stride = 1
    padding = 0
    groups = 4
    drtam = False
    refinement = False
    store_imgs = False

if config == 2:
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



    local_kernel_size
    encoder_arch
    resultdir
    dataset 
    attn_padding = 0
    attn_groups = 4
    attn_stride = 1
    multigpu
    checkpointdir
    dir_img 
    store_imgs 