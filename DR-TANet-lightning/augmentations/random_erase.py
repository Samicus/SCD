import numpy as np
import cv2

def random_erase_augment(img_t0, img_t1, img_mask, WIDTH_DIV=2.0, HEIGHT_DIV=2.0):
    
    w, h, _ = img_t0.shape
    
    WIDTH_THRESHOLD = w / WIDTH_DIV
    HEIGHT_THRESHOLD = h / HEIGHT_DIV
    
    # Random width and height of erased area
    erase_width = np.random.randint(WIDTH_THRESHOLD / 2, WIDTH_THRESHOLD)
    erase_height = np.random.randint(HEIGHT_THRESHOLD / 2, HEIGHT_THRESHOLD)
    
    # Random horizontal positioning of erased area
    x_pos_left = np.random.randint(0, w - erase_width)
    x_pos_right = x_pos_left + erase_width
    
    # Random vertical positioning of erased area
    y_pos_down = np.random.randint(0, h - erase_height)
    y_pos_up = y_pos_down + erase_height
    
    #print(img_mask.shape)
    #exit()
    
    # Erasing same random area in t0, t1 and mask.
    img_t0[x_pos_left : x_pos_right, y_pos_down : y_pos_up, :] = 128
    img_t1[x_pos_left : x_pos_right, y_pos_down : y_pos_up, :] = 128
    img_mask[x_pos_left : x_pos_right, y_pos_down : y_pos_up] = 255
    
    return img_t0, img_t1, img_mask


# FOR DEBUG
if __name__ == '__main__':
    img_t0 = cv2.imread("/home/arwin/Documents/git/PCD/TSUNAMI/set0/test/t0/00000080.jpg", 1)
    img_t1 = cv2.imread("/home/arwin/Documents/git/PCD/TSUNAMI/set0/test/t0/00000080.jpg", 1)
    img_mask = cv2.imread("/home/arwin/Documents/git/PCD/TSUNAMI/set0/test/mask/00000080.bmp", 0)
    
    img_t0_erased, img_t1_erased, img_mask_erased = random_erase(img_t0, img_t1, img_mask)
    
    # Using cv2.imshow() method 
    # Displaying the image 
    cv2.imshow("Random Erase", img_mask_erased)
    
    #waits for user to press any key 
    #(this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0) 
    
    #closing all open windows 
    cv2.destroyAllWindows()