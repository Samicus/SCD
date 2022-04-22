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

        """
        Input:
            scale
            rotation

        load new imagepair and mask to copy and paste
        resize images -  uniform(1-scale, 1)
        rotate uniform(-rotation rotation)
        
        x1,y1 <--  get random x,y position on original image to paste
        x2 <--  x1 + copy image width
        y2 <-- y1 + copy image width

        original_mask <-- copy mask
        inverted_mask <-- inverted copy mask
        
        img_t0[y1:y2, x1:x2] = (copy_t0 * invert_mask) + (img_t0[y1:y2, x1:x2] * original_mask)
        img_t1[y1:y2, x1:x2] = (copy_t1 * invert_mask) + (img_t1[y1:y2, x1:x2] * original_mask)
        img_mask[y1:y2, x1:x2] = (copy_mask * invert_mask) + (img_mask[y1:y2, x1:x2] * original_mask)
        """


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
        