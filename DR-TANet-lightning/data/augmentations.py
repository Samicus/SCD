from random import random, choices, shuffle, uniform
import numpy as np
import cv2
import math
import os
from os.path import join as pjoin, splitext as spt
from PIL import Image
import albumentations as A
from util import load_config


class DataAugment:
    def __init__(self, t0_root, t1_root, mask_root, filename, augmentations, aug_params,shape=(256, 256), trial=None):
        self.t0_root = t0_root
        self.t1_root = t1_root
        self.mask_root = mask_root
        self.filename = filename
        self.index = None
        self.shape = shape

        random_shadow_p = .5
        color_jitter_p = .5
        if trial:
            random_shadow_p = trial.suggest_uniform("random_shadow_threshold", 0.0, 1.0)
            color_jitter_p = trial.suggest_uniform("color_jitter_threshold", 0.0, 1.0)
        
        self.transform1 = A.Compose([
                A.RandomShadow(p=random_shadow_p),
                A.ColorJitter(p=color_jitter_p)
                ])
        
        self.transform2 = A.Compose([
            A.HorizontalFlip(p=1.),
            ])

        # Augmentation switches
        self.copy_paste_on = augmentations["copy_paste_on"]
        self.random_erase_on = augmentations["random_erase_on"]
        self.mosaic_on = augmentations["mosaic_on"]
        self.albumentations_on = augmentations["albumentations_on"]

        # Load Augmentation Parameters
        
        mosaic_params = aug_params["MOSAIC"]
        random_erase_params = aug_params["RANDOM_ERASE"]
        copy_paste_params = aug_params["COPY_PASTE"]
        
        if trial:
            # MOSAIC
            self.mosaic_th = trial.suggest_uniform("mosaic_threshold", 0.0, 1.0)
            self.translate = trial.suggest_uniform("mosaic_translate", 0.0, 1.0)
            self.scale = trial.suggest_uniform("mosaic_scale", 0.0, 0.99)
            self.rotation = trial.suggest_int("mosaic_rotation", 0, 90)
            
            # Random Erase
            self.random_erase_th = trial.suggest_uniform("random_erase_threshold", 0.0, 1.0)

            # Copy Paste
            copy_paste_scale = trial.suggest_uniform("copy_paste_scale", 0.0, 0.99)
            copy_paste_rotation = trial.suggest_int("copy_paste_rotation", 0, 180)
            self.copy_paste_th = trial.suggest_uniform("copy_paste_threshold", 0.0, 1.0)
        else:
            # MOSAIC
            self.mosaic_th = mosaic_params["mosaic_th"]
            mosaic_translate = mosaic_params["translate"]
            mosaic_scale = mosaic_params["scale"]
            mosaic_rotation = mosaic_params["rotation"]
            
            # Random Erase
            self.random_erase_th = random_erase_params["random_erase_th"]

            # Copy Paste
            copy_paste_scale = copy_paste_params["copy_paste_scale"]
            copy_paste_rotation = copy_paste_params["copy_paste_rotation"]
            self.copy_paste_th = copy_paste_params["copy_paste_th"]
        
        self.copy_paste = CopyPaste(t0_root, t1_root, mask_root, filename, scale=copy_paste_scale, rotation=copy_paste_rotation)
        self.mosaic = Mosaic(mosaic_rotation, mosaic_translate, shape, mosaic_scale)

    def __call__(self, index):
        """
        albumentation -> random_crop -> mosaic
        """
        self.index = index
        self.img_t0_list = []
        self.img_t1_list = []
        self.img_mask_list = []
        
        mosaic_rand = random()
        run_mosaic = (self.mosaic_on) and (mosaic_rand < self.mosaic_th)
        
        # if mosaic -> load 4 images into list and perform the other augmentations on them individually
        if run_mosaic:
            self.load_mosaic_imgs()
        else:
            self.load_current_img()
        
        #Apply Copy-Paste 
        if self.copy_paste_on:
            self.apply_copy_paste()

        if self.albumentations_on:
            # Augments data with albumentations, Which augments to be applied is chosen in params.py by albumentations_config
            self.albumentation_augment()

        # Apply random erase
        if self.random_erase_on:
            self.random_erase_augment(WIDTH_DIV=2.0, HEIGHT_DIV=2.0)

        # Apply MOSAIC
        if run_mosaic:
            img_t0, img_t1, img_mask = self.mosaic(self.img_t0_list, self.img_t1_list, self.img_mask_list)
        else:
            img_t0, img_t1, img_mask = self.img_t0_list[0], self.img_t1_list[0], self.img_mask_list[0]

        return img_t0, img_t1, img_mask

    def apply_copy_paste(self):
        for i in range(len(self.img_t0_list)):
            if random() < self.copy_paste_th:
                self.img_t0_list[i], self.img_t1_list[i], self.img_mask_list[i] = self.copy_paste(self.img_t0_list[i], 
                                                                                                  self.img_t1_list[i], 
                                                                                                  self.img_mask_list[i])

    def load_mosaic_imgs(self):
            """
            Loads current images t0, t1, mask with 3 other random images into lists
            """
            # load 3 new images into lists
            nr_images = len(self.filename)
            indices = range(nr_images-1) 
            indices = [self.index] + choices(indices, k=3) 
            shuffle(indices)

            for idx in indices:
                fn = self.filename[idx]

                fn_t0 =   pjoin(self.t0_root,fn+'.jpg')
                fn_t1 =   pjoin(self.t1_root,fn+'.jpg')
                fn_mask = pjoin(self.mask_root,fn+'.bmp')

                img_t0 =   cv2.imread(fn_t0, 1)
                img_t1 =   cv2.imread(fn_t1, 1)
                img_mask = cv2.imread(fn_mask, 0)
                #img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)

                self.img_t0_list.append(img_t0)
                self.img_t1_list.append(img_t1)
                self.img_mask_list.append(img_mask)

    def load_current_img(self):
        """
        loads current images t0, t1 and mask into lists with size 1
        """
        fn = self.filename[self.index]

        fn_t0 =   pjoin(self.t0_root,fn+'.jpg')
        fn_t1 =   pjoin(self.t1_root,fn+'.jpg')
        fn_mask = pjoin(self.mask_root,fn+'.bmp')

        img_t0 =   cv2.imread(fn_t0, 1)
        img_t1 =   cv2.imread(fn_t1, 1)
        img_mask = cv2.imread(fn_mask, 0)
        #img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)

        self.img_t0_list.append(img_t0)
        self.img_t1_list.append(img_t1)
        self.img_mask_list.append(img_mask)


    def random_erase_augment(self, WIDTH_DIV=2.0, HEIGHT_DIV=2.0):
        
        w, h, _ = self.img_t0_list[0].shape
        
        WIDTH_THRESHOLD = w / WIDTH_DIV
        HEIGHT_THRESHOLD = h / HEIGHT_DIV
        for i in range(len(self.img_t0_list)):
            if random() < self.random_erase_th:
                # Random width and height of erased area
                erase_width = np.random.randint(WIDTH_THRESHOLD / 2, WIDTH_THRESHOLD)
                erase_height = np.random.randint(HEIGHT_THRESHOLD / 2, HEIGHT_THRESHOLD)
                
                # Random horizontal positioning of erased area
                x_pos_left = np.random.randint(0, w - erase_width)
                x_pos_right = x_pos_left + erase_width
                
                # Random vertical positioning of erased area
                y_pos_down = np.random.randint(0, h - erase_height)
                y_pos_up = y_pos_down + erase_height
                
                # Erasing same random area in t0, t1 and mask.
                self.img_t0_list[i][x_pos_left : x_pos_right, y_pos_down : y_pos_up, :] = 128
                self.img_t1_list[i][x_pos_left : x_pos_right, y_pos_down : y_pos_up, :] = 128
                self.img_mask_list[i][x_pos_left : x_pos_right, y_pos_down : y_pos_up] = 255

    def albumentation_augment(self):

        for i, (t0_img, t1_img, mask_img) in enumerate(list(zip(self.img_t0_list, self.img_t1_list, self.img_mask_list))):
                t0_img = np.array(t0_img)
                # apply random shadow/channel shuffle randomly for t0 and t1
                transformed_t0 = self.transform1(image=t0_img)["image"]
                t1_img = np.array(t1_img)
                transformed_t1 = self.transform1(image=t1_img)["image"]

                # Horizontal flip on t0,t1,mask
                if random() > .5:
                    transformed_t0 = self.transform2(image=transformed_t0)["image"]
                    transformed_t1 = self.transform2(image=transformed_t1)["image"]
                    transformed_mask = self.transform2(image=mask_img)["image"]

                    self.img_mask_list[i] = transformed_mask

                self.img_t0_list[i] = transformed_t0
                self.img_t1_list[i] = transformed_t1

class Mosaic():
    def __init__(self, rotation, translate, shape, scale):
        self.rotation = rotation
        self.translate = translate
        self.shape = shape
        self.scale = scale

    def __call__(self, img_t0_list, img_t1_list, img_mask_list):
        """
        Description:
            - This method augments an image using the mosaic augmentation method. It combines 4 images 
            by translating and rescaling without changing the aspect ratio. Additional 
            augmentations such as rotation, shear and perspective can be applied. The hyper parameters
            for the augmentations can be found in params.py.
        Output:
            - img4_t0, img4_t1, img4_mask:
                    The output is the augmented images t0, t1 and mask as 
                    numpy ndarrays with the same output size as the original. 
            
        """
        
        img_shape = self.shape
        
        s = img_shape[0]            
        s1 = img_shape[1]        
        mosaic_border = [-s//2 , -s1//2]
        yc = img_shape[0]
        xc = img_shape[1]
        
        for i in range(4):

            img_t0 =   img_t0_list[i]
            img_t1 =   img_t1_list[i]
            img_mask = img_mask_list[i]

            # scale each image seperately
            resize_factor = uniform(1.0 - self.scale, 1.0 + self.scale)
            h_orig, w_orig, _ = img_t0.shape
            img_t0 =   cv2.resize(img_t0,   (int(resize_factor*w_orig), int(resize_factor*h_orig)))
            img_t1 =   cv2.resize(img_t1,   (int(resize_factor*w_orig), int(resize_factor*h_orig)))
            img_mask = cv2.resize(img_mask, (int(resize_factor*w_orig), int(resize_factor*h_orig)))
    
            h, w, _ = img_t0.shape
            
            # place img in img4
            if i == 0:  # top left
                img4_t0 = np.full((s * 2, s1 * 2, img_t0.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                img4_t1 = np.full((s * 2, s1 * 2, img_t1.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                img4_mask = np.full((s * 2, s1 * 2), 255, dtype=np.uint8)                   # base image with 4 tiles

                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s1 * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s1 * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4_t0[y1a:y2a, x1a:x2a] = img_t0[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            img4_t1[y1a:y2a, x1a:x2a] = img_t1[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            img4_mask[y1a:y2a, x1a:x2a] = img_mask[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        
        img4_t0, img4_t1, img4_mask = self.random_perspective(img4_t0, img4_t1, img4_mask,degrees=self.rotation, translate=self.translate,  border=mosaic_border)
        
        return img4_t0, img4_t1, img4_mask


    def random_perspective(self, im_t0, im_t1, im_mask, degrees=0, translate=.2, border=(0, 0)):
        
        height = im_t0.shape[0] + border[0] * 2  # shape(h,w,c)
        width = im_t0.shape[1] + border[1] * 2
    
        # Center
        C = np.eye(3)
        C[0, 2] = -im_t0.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -im_t0.shape[0] / 2  # y translation (pixels)

        # Rotation and Scale
        R = np.eye(3)
        a = uniform(-degrees, degrees)
        if (not degrees == 0) and (random() >= .5):
            a += 180
        s = 1.0
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ R @ C  # order of operations is important
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed

            im_t0 = cv2.warpAffine(im_t0, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            im_t1 = cv2.warpAffine(im_t1, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            im_mask = cv2.warpAffine(im_mask, M[:2], dsize=(width, height), borderValue=(255, 255, 255))

        return im_t0, im_t1, im_mask

class CopyPaste():
    ''' 
    Copies, resizes and rotates masked part of a random image and 
    pastes it on a random section of the input image image 
    '''


    def __init__(self, t0_root, t1_root, mask_root, filename, scale, rotation=180):
        """
        params: 
            t0_root: root directory to t0 images
            t1_root: root directory to t1 images
            mask_root: root directory to mask with .png files
            filename: a list of filenames in mask directory
            scale: scaling of the pasted image randomly between scale[0] and scale[1]
            rotation: rotation of the pasted image randomly between -rotation and rotation in degrees
        """
        self.scale = scale
        self.t0_root = t0_root
        self.t1_root = t1_root
        self.mask_root = mask_root
        self.filename = filename
        self.rotation = rotation
    def __call__(self, img_t0, img_t1, img_mask):
        copy_t0, copy_t1, copy_mask = self.load_random_imgs()
    

        h, w, _ = img_t0.shape

        #copy_t0 = transforms.Resize(size=(H, W))(self.instance)
        resize_factor = uniform(1.0 - self.scale, 1.0)
        resize_factor = max(0.01, resize_factor)
        copy_t0 =   cv2.resize(copy_t0,   (int(resize_factor*w), int(resize_factor*h)))
        copy_t1 =   cv2.resize(copy_t1,   (int(resize_factor*w), int(resize_factor*h)))
        copy_mask = cv2.resize(copy_mask, (int(resize_factor*w), int(resize_factor*h)))
        random_rotation = uniform(-self.rotation, self.rotation)
        copy_t0 = self.rotate_image(copy_t0, random_rotation)
        copy_t1 = self.rotate_image(copy_t1, random_rotation)
        
        copy_mask = self.rotate_image(copy_mask, random_rotation)
        h_resized, w_resized, _ = copy_t0.shape

        #Convert to np arrays
        copy_t0 = np.asarray(copy_t0)
        copy_t1 = np.asarray(copy_t1)
        copy_mask = np.asarray(copy_mask)

        img_t0 = np.asarray(img_t0)
        img_t1 = np.asarray(img_t1)
        img_mask = np.asarray(img_mask)

        #Random positions for pasted images
        h_start = np.random.randint(0, max(1, h - h_resized))
        w_start = np.random.randint(0, max(1, w - w_resized))
        y1, y2 = h_start, h_start + h_resized
        x1, x2 = w_start, w_start + w_resized

        binary_mask = 1.0 * (copy_mask > 0)
        invert_mask = 1.0 * (np.logical_not(binary_mask).astype(int))

        # expand dimensions of mask to match rgb images
        blur_invert_mask = np.expand_dims(invert_mask, 2)  # Expanding dims to match channels
        blur_binary_mask = np.expand_dims(binary_mask, 2)
        
        # Paste image
        img_t0[y1:y2, x1:x2] = (copy_t0 * blur_invert_mask) + (img_t0[y1:y2, x1:x2] * blur_binary_mask)
        img_t1[y1:y2, x1:x2] = (copy_t1 * blur_invert_mask) + (img_t1[y1:y2, x1:x2] * blur_binary_mask)
        img_mask[y1:y2, x1:x2] = (copy_mask * invert_mask) + (img_mask[y1:y2, x1:x2] * binary_mask)

        return img_t0, img_t1, img_mask


    def rotate_image(self, image, angle):
        """
        Rotates an image 
        """
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rotated_img = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        return rotated_img

    def load_random_imgs(self):
        """
        returns random images t0, t1 and mask
        """
        nr_images = len(self.filename)
        index = np.random.randint(0, nr_images) 
        fn = self.filename[index]

        fn_t0 =   pjoin(self.t0_root,fn+'.jpg')
        fn_t1 =   pjoin(self.t1_root,fn+'.jpg')
        fn_mask = pjoin(self.mask_root,fn+'.bmp')

        img_t0 =   cv2.imread(fn_t0, 1)
        img_t1 =   cv2.imread(fn_t1, 1)
        img_mask = cv2.imread(fn_mask, 0)
        #img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)
        return img_t0, img_t1, img_mask
        
# DEBUG
if __name__ == '__main__':
    
    dirname = os.path.dirname
    #root = pjoin(dirname(dirname(dirname(dirname(__file__)))), "TSUNAMI/set0/train")
    root = '/home/samnehme/Dev/SCD_project/TSUNAMI/set0/train'
    img_t0_root = pjoin(root,'t0')
    img_t1_root = pjoin(root,'t1')
    img_mask_root = pjoin(root,'mask/bmp')
    filename = list(spt(f)[0] for f in os.listdir(img_mask_root))

    config_path = "DR-TANet-lightning/config/augmentations/M_CP_RE_A.yaml"
    augmentations = load_config(config_path)["RUN"]
    aug_params = load_config("DR-TANet-lightning/config/augparams.yaml")
    
    data_augmenter = DataAugment(img_t0_root, img_t1_root, img_mask_root, filename, augmentations, aug_params, shape=(224, 1024))

    index = 18
    img_t0, img_t1, img_mask = data_augmenter(index)


    folder_path = '/home/samnehme/Dev/SCD_project/SCD/DR-TANet-lightning/images'

    #img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
    #img_t0 = Image.fromarray(img_t0.astype(np.uint8))
    #img_t0.show()
    cv2.imwrite(folder_path+'/t0.jpg', img_t0)
    #img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
    #img_t1 = Image.fromarray(img_t1.astype(np.uint8))
    #img_t1.show()
    cv2.imwrite(folder_path+'/t1.jpg', img_t1)
    #img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
    #img_mask = Image.fromarray(img_mask.astype(np.uint8))
    #img_mask.show()
    cv2.imwrite(folder_path+'/mask.jpg', img_mask)
    #cv2.waitKey(0)