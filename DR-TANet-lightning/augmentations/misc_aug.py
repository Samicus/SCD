import cv2
import albumentations as A
import numpy as np

# FOR DEBUG
if __name__ == '__main__':
    img_t0 = cv2.imread("/home/arwin/Documents/git/PCD/TSUNAMI/set0/test/t0/00000080.jpg", 1)
    img_t1 = cv2.imread("/home/arwin/Documents/git/PCD/TSUNAMI/set0/test/t0/00000080.jpg", 1)
    img_mask = cv2.imread("/home/arwin/Documents/git/PCD/TSUNAMI/set0/test/mask/00000080.bmp", 0)
    
    h, w = img_mask.shape
    
    transform_img = A.Compose([A.RandomCrop(width=int(w/2), height=h),
                           A.HorizontalFlip(p=0.5),
                           A.RandomBrightnessContrast(p=1),
                           A.RandomShadow(p=1)
    ])
        
    transformed_img_t0_1 = transform_img(image=img_t0)
    transformed_img_t0_2 = transform_img(image=img_t0)
    
    transformed_imgs_t0 = np.hstack((transformed_img_t0_1['image'], transformed_img_t0_2['image']))
    
    disp_img = np.vstack((img_t0, transformed_imgs_t0))
    
    # Using cv2.imshow() method 
    # Displaying the image 
    cv2.imshow("Augmentation", disp_img)
    
    #waits for user to press any key 
    #(this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0) 
    
    #closing all open windows 
    cv2.destroyAllWindows()