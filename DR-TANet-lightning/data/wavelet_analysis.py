from glob import glob
import os
import matplotlib
import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image, ImageOps
from pytorch_wavelets import DWTForward, DWTInverse
from matplotlib import pyplot as plt
import argparse
from os.path import join as pjoin

matplotlib.use("TKAgg")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True,
    help="path to dataset")
parser.add_argument("--mask", action="store_true")
parser.add_argument("--zero_high", action="store_true")
parser.add_argument("--zero_low", action="store_true")
parser.add_argument("--LH", action="store_false")
parser.add_argument("--HL", action="store_false")
parser.add_argument("--HH", action="store_false")
parsed_args = parser.parse_args()

INPUT_PATH = parsed_args.input
DECOMPOSITION_LEVELS = 7
DATASET_FOLDERS = ["t0", "t1"]
HIGH_PASS_LABELS = ["LH", "HL", "HH"]
ZERO_HIGH_PASS = [parsed_args.LH, parsed_args.HL, parsed_args.HH]
HIGH_PASS_PLOT_DIR = "high_pass_plots"
DATASET_NAME = "PCD"

if "TSUNAMI" in INPUT_PATH:
    DATASET_NAME = "TSUNAMI"
elif "GSV" in INPUT_PATH:
    DATASET_NAME = "GSV"

COMPRESSED_IMG_DIR = pjoin("compressed_images", DATASET_NAME)

def plot_high_pass_coeffs(coefficients_list, image_name):
    
    os.makedirs(HIGH_PASS_PLOT_DIR, exist_ok=True)
    
    for _, (high_pass, high_pass_label) in enumerate(zip(coefficients_list, HIGH_PASS_LABELS)):
            
        high_pass = torch.flatten(high_pass)
        pixel_steps = torch.linspace(0, len(high_pass), steps=len(high_pass))

        plt.figure()
        # TkAgg backend
        plt.plot(pixel_steps, abs(high_pass))
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.savefig("high_pass_plots{}{}_{}".format(os.sep, image_name, high_pass_label), bbox_inches='tight')
        
def zero_high_pass_coeffs(high_pass_coeff_list, decomposition_levels):
    for decomposition_level in range(decomposition_levels):
        for i, _ in enumerate(high_pass_coeff_list[decomposition_level][0, 0, :]):
            if ZERO_HIGH_PASS[i]:
                high_pass_coeff_list[decomposition_level][0, 0, i] = torch.zeros(size=high_pass_coeff_list[decomposition_level][0, 0, i].size())
    return high_pass_coeff_list

def wavelet_compress(decomposition_levels):
    
    # Wavelet transform objects
    dwt = DWTForward(J=decomposition_levels, # level of the transform 
                        wave='haar', # wavelet function
                        #mode='symmetric'
                        ) # padding at edges
    idwt = DWTInverse(wave='haar',
                      #mode='symmetric'
                      )
    
    original_byte_size = {}
    compressed_byte_size = {}
    
    print("Compression ratios for {}".format(DATASET_NAME))
    
    file_ext = "*.jpg"
    sub_directories = DATASET_FOLDERS
    if parsed_args.mask:
        file_ext = "*.bmp"
        sub_directories = ["mask"]
    
    for sub_dir in sub_directories:
        
        current_dir = pjoin(INPUT_PATH, sub_dir)
        
        original_byte_size[sub_dir] = 0.0
        compressed_byte_size[sub_dir] = 0.0
        
        for img_path in glob(pjoin(current_dir, file_ext)):
            
            # Save byte size of original image
            original_byte_size[sub_dir] += os.stat(img_path).st_size
            
            # Open and preprocess image
            img = Image.open(img_path)
            img = ImageOps.grayscale(img)   # To grayscale
            img_name = img_path.split("/")[-1]
            torch_img = ToTensor()(img) # note! funky syntax
            torch_img = torch.unsqueeze(torch_img[0,:,:], dim=0) # Remove excess channels
            torch_img = torch.unsqueeze(torch_img, dim=0) # add batch dimension for DWT

            # Transformation
            low_pass_coeff, high_pass_coeff_list = dwt(torch_img)
            
            if parsed_args.zero_high:
                # Set high-pass coefficients to zero
                high_pass_coeff_list = zero_high_pass_coeffs(high_pass_coeff_list, decomposition_levels)
            if parsed_args.zero_low:
                # Set low-pass coefficients to zero
                low_pass_coeff = torch.zeros(size=low_pass_coeff.size())
                
            # Inverse Discrete Wavelet Transform
            tensor_compressed_image = idwt((low_pass_coeff, high_pass_coeff_list))
            
            # Save compressed image
            tensor_compressed_image = torch.squeeze(tensor_compressed_image, dim=0) # remove batch dimension
            compressed_image = ToPILImage()(tensor_compressed_image)
            compressed_sub_dir = pjoin(COMPRESSED_IMG_DIR, sub_dir)
            for i, high_pass_label in enumerate(HIGH_PASS_LABELS):
                if not ZERO_HIGH_PASS[i]:
                    compressed_sub_dir += "_{}".format(high_pass_label)
            os.makedirs(compressed_sub_dir, exist_ok=True)
            compressed_img_path = pjoin(compressed_sub_dir, img_name)
            compressed_image.save(compressed_img_path)
            
            # Save byte size of compressed image
            compressed_byte_size[sub_dir] += os.stat(compressed_img_path).st_size
            
        print("R_{} = {}".format(sub_dir, original_byte_size[sub_dir] / compressed_byte_size[sub_dir]))
        
        
if __name__ == '__main__':
    wavelet_compress(DECOMPOSITION_LEVELS)
