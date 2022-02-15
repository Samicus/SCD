import random
import numpy as np
import cv2
import os
import glob
import numpy as np
from os.path import join as pjoin, splitext as spt
from PIL import Image

OUTPUT_SIZE = (224, 1024)  # Height, Width
SCALE_RANGE = (0.3, 0.7)
FILTER_TINY_SCALE = 1 / 50  # if height or width lower than this scale, drop it.

ANNO_DIR = 'dataset/WiderPerson/Annotations/'
IMG_DIR = 'dataset/WiderPerson/Images/'

category_name = ['background', 'person']


def mosaic_augment(t0, t1, mask, filename, t0_root, t1_root, mask_root):


    nr_images = len(filename)
    random_indices = random.sample(range(0,nr_images), 3)
    
    fn1 = filename[random_indices[0]]
    fn1_t0 = pjoin(t0_root,fn1+'.jpg')
    fn1_t1 = pjoin(t1_root,fn1+'.jpg')
    fn1_mask = pjoin(mask_root,fn1+'.png')
    
    fn2 = filename[random_indices[1]]
    fn2_t0 = pjoin(t0_root,fn2+'.jpg')
    fn2_t1 = pjoin(t1_root,fn2+'.jpg')
    fn2_mask = pjoin(mask_root,fn2+'.png')
    
    fn3 = filename[random_indices[2]]
    fn3_t0 = pjoin(t0_root,fn3+'.jpg')
    fn3_t1 = pjoin(t1_root,fn3+'.jpg')
    fn3_mask = pjoin(mask_root,fn3+'.png')


    img1_t0 = cv2.imread(fn1_t0, 1)
    img1_t1 = cv2.imread(fn1_t1, 1)
    mask1 = cv2.imread(fn1_mask, 0)
    
    img2_t0 = cv2.imread(fn2_t0, 1)
    img2_t1 = cv2.imread(fn2_t1, 1)
    mask2 = cv2.imread(fn2_mask, 0)
    
    img3_t0 = cv2.imread(fn3_t0, 1)
    img3_t1 = cv2.imread(fn3_t1, 1)
    mask3 = cv2.imread(fn3_mask, 0)

    all_t0_list = [t0, img1_t0, img2_t0, img3_t0]
    all_t1_list = [t1, img1_t1, img2_t1, img3_t1]
    all_mask_list = [mask, mask1, mask2, mask3]

    output_img_t0, output_img_t1, output_img_mask = update_image_and_anno(all_t0_list, all_t1_list, all_mask_list,
                                                 OUTPUT_SIZE, SCALE_RANGE,
                                                 filter_scale=FILTER_TINY_SCALE)
    
    output_img_t0 = cv2.cvtColor(output_img_t0, cv2.COLOR_BGR2RGB)
    output_img_t0 = Image.fromarray(output_img_t0.astype(np.uint8))
    output_img_t0.show()
    
    output_img_t1 = cv2.cvtColor(output_img_t1, cv2.COLOR_BGR2RGB)
    output_img_t1 = Image.fromarray(output_img_t1.astype(np.uint8))
    output_img_t1.show()
    
    output_img_mask = cv2.cvtColor(output_img_mask, cv2.COLOR_BGR2RGB)
    output_img_mask = Image.fromarray(output_img_mask.astype(np.uint8))
    output_img_mask.show()
"""
    cv2.imwrite('img/output.jpg', new_image)
    for anno in new_annos:
        start_point = (int(anno[1] * OUTPUT_SIZE[1]), int(anno[2] * OUTPUT_SIZE[0]))
        end_point = (int(anno[3] * OUTPUT_SIZE[1]), int(anno[4] * OUTPUT_SIZE[0]))
        cv2.rectangle(new_image, start_point, end_point, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite('img/output_box.jpg', new_image)

    
    """
    

def update_image_and_anno(all_t0_list, all_t1_list, all_mask_list, output_size, scale_range, filter_scale=0.):
    output_img_t0 = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    output_img_t1 = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    output_img_mask = np.zeros([output_size[0], output_size[1]], dtype=np.uint8)
    
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])
    print(divid_point_x)
    print(divid_point_y)
    for i, _ in enumerate(all_t0_list):
        img_t0 = all_t0_list[i]
        img_t1 = all_t1_list[i]
        img_mask = all_mask_list[i]
        
        if i == 0:  # top-left
            # t0
            img_t0 = cv2.resize(img_t0, (divid_point_x, divid_point_y))
            output_img_t0[:divid_point_y, :divid_point_x, :] = img_t0
            # t1
            img_t1 = cv2.resize(img_t1, (divid_point_x, divid_point_y))
            output_img_t1[:divid_point_y, :divid_point_x, :] = img_t1
            # mask
            img_mask = cv2.resize(img_mask, (divid_point_x, divid_point_y))
            output_img_mask[:divid_point_y, :divid_point_x] = img_mask
        elif i == 1:  # top-right
            # t0
            img_t0 = cv2.resize(img_t0, (output_size[1] - divid_point_x, divid_point_y))
            output_img_t0[:divid_point_y, divid_point_x:output_size[1], :] = img_t0
            # t1
            img_t1 = cv2.resize(img_t1, (output_size[1] - divid_point_x, divid_point_y))
            output_img_t1[:divid_point_y, divid_point_x:output_size[1], :] = img_t1
            # mask
            img_mask = cv2.resize(img_mask, (output_size[1] - divid_point_x, divid_point_y))
            output_img_mask[:divid_point_y, divid_point_x:output_size[1]] = img_mask

        elif i == 2:  # bottom-left
            # t0
            img_t0 = cv2.resize(img_t0, (divid_point_x, output_size[0] - divid_point_y))
            output_img_t0[divid_point_y:output_size[0], :divid_point_x, :] = img_t0
            # t1
            img_t1 = cv2.resize(img_t1, (divid_point_x, output_size[0] - divid_point_y))
            output_img_t1[divid_point_y:output_size[0], :divid_point_x, :] = img_t1
            # mask
            img_mask = cv2.resize(img_mask, (divid_point_x, output_size[0] - divid_point_y))
            output_img_mask[divid_point_y:output_size[0], :divid_point_x] = img_mask
        else:  # bottom-right
            # t0
            img_t0 = cv2.resize(img_t0, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
            output_img_t0[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img_t0
            # t1
            img_t1 = cv2.resize(img_t1, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
            output_img_t1[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img_t1
            # mask
            img_mask = cv2.resize(img_mask, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
            output_img_mask[divid_point_y:output_size[0], divid_point_x:output_size[1]] = img_mask

    return output_img_t0, output_img_t1, output_img_mask



if __name__ == '__main__':
    dirname = os.path.dirname
    root = pjoin(dirname(dirname(dirname(__file__))), "TSUNAMI/set0/train")
    img_t0_root = pjoin(root,'t0')
    img_t1_root = pjoin(root,'t1')
    img_mask_root = pjoin(root,'mask')
    filename = list(spt(f)[0] for f in os.listdir(img_mask_root))
    
    fn = filename[8]
    fn_t0 = pjoin(img_t0_root,fn+'.jpg')
    fn_t1 = pjoin(img_t1_root,fn+'.jpg')
    fn_mask = pjoin(img_mask_root,fn+'.png')
    
    t0 = cv2.imread(fn_t0, 1)
    t1 = cv2.imread(fn_t1, 1)
    mask = cv2.imread(fn_mask, 0)
    mosaic_augment(t0, t1, mask, filename, img_t0_root, img_t1_root, img_mask_root)


    main()