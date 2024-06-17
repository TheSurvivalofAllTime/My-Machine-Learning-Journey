# Author: Nikolin Prenga

# This Python module is designed for image preprocessing tailored for a machine learning context.

#It includes functionality to load and preprocess images from a specified directory, applying 
#various image processing techniques.

# Key operations include converting images to grayscale, resizing, thresholding, and extracting significant contours.

#Additionally, the module handles image data preparation for model training by 
#normalizing image sizes and associating images with labels based on filename prefixes.

#It also offers functionality to analyze image properties such as size and aspect ratio, 
#which aids in understanding dataset characteristics.

import warnings
import numpy as np
import os
import cv2 
warnings.filterwarnings("ignore")



class Image_Processing(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.label_mapping = {
            'g': 1,  # glioma_tumor
            'm': 2,  # meningioma_tumor
            'n': 0,  # no_tumor
            'p': 3   # pituitary_tumor
        }
    # Assuming 'extract_contour' is a function that crops the image to the largest found contour
    def extract_contour(self, image):
        self.image = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #blurred = cv2.GaussianBlur(gray, (5, 5), 2)
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return image[y:y+h, x:x+w]
        
        return image  # Return original if no contours found



    def load_data_and_crop(self):
        np.random.seed(42)
        labels_filenames = []
        label_numeric = []
        images_save = []
        for filename in os.listdir(self.dir_name):
            img_path = os.path.join(self.dir_name, filename)
            self.image = cv2.imread(img_path)

            if self.image is None:
                print("Failed to load image:", img_path)  
                continue

            cropped_image = self.extract_contour(self.image)
            cropped_image=cv2.resize(cropped_image, (224,224))
            images_save.append(cropped_image)

            prefix = filename[0]
            label_number = self.label_mapping.get(prefix, -1)  
            label_numeric.append(label_number)
            labels_filenames.append(filename)

        num_samples = len(images_save)
        # Generate shuffled indices
        shuffled_test_indices = np.random.permutation(num_samples)
        data = np.array(images_save)[shuffled_test_indices]
        labels =  np.array(label_numeric)[shuffled_test_indices]
        filenames_labels = np.array(labels_filenames)[shuffled_test_indices]

        return  data, labels, filenames_labels
    

    def load_data_and_crop_Optionally(self):
        labels_filenames = []
        label_numeric = []
        images_save = []
        for filename in os.listdir(self.dir_name):
            img_path = os.path.join(self.dir_name, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
             #blurred = cv2.GaussianBlur(gray, (5, 5), 2)
            #_, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
            #_, thresh = cv2.threshold(blurred, 48, 300, cv2.THRESH_BINARY)
            _, thresh = cv2.threshold(image, 48, 300, cv2.THRESH_BINARY)
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cropped_image = image[y:y+h, x:x+w]
                reshaped_image = cv2.resize(cropped_image, (240, 240))
                images_save.append(reshaped_image)

                prefix = filename[0]
                label_number = self.label_mapping.get(prefix, -1)  # Default to -1 if prefix not found
                label_numeric.append(label_number)
                labels_filenames.append(filename)
        np.random.seed(42)

        num_samples = len(images_save)
        # Generate shuffled indices
        shuffled_test_indices = np.random.permutation(num_samples)
        data = np.array(images_save)[shuffled_test_indices]
        labels =  np.array(label_numeric)[shuffled_test_indices]
        filenames_labels = np.array(labels_filenames)[shuffled_test_indices]

        return data, labels, filenames_labels
    
    def Check_size_shape(self):
        num_samples =0
        num_nonsquares =0
        num_under250 = 0
        num_under300 = 0
        num_under400 = 0
        num_under500 = 0
        num_under600 = 0
        above600 =0
        for filenames in os.listdir(self.dir_name):
            img_path = os.path.join(self.dir_name, filenames)
            ims = cv2.imread(img_path)
            num_samples = num_samples+ 1
            if ims is None:
                continue

            if ims.shape[0] != ims.shape[1]:
                num_nonsquares = num_nonsquares+1

            else:
                if  ims.shape[0] <250:
                    num_under250 = num_under250+1

                elif   250 <= ims.shape[0] < 300:
                    num_under300 = num_under300+1

                elif 300 <= ims.shape[0] < 400:
                    num_under400 = num_under400 +1

                elif 400 <= ims.shape[0] < 500:
                    num_under500 = num_under500+1

                elif 500 <= ims.shape[0] < 600:
                    num_under600= num_under600 +1
                else:
                    above600 = above600 +1
        print(f'Total number of samples analyzed: {num_samples}')
        print(f'Number of non-square images (different width and height): {num_nonsquares}')
        print(f'Number of square images smaller than 250x250 pixels: {num_under250}')
        print(f'Number of square images sized between 250x250 and 299x299 pixels: {num_under300}')
        print(f'Number of square images sized between 300x300 and 399x399 pixels: {num_under400}')
        print(f'Number of square images sized between 400x400 and 499x499 pixels: {num_under500}')
        print(f'Number of square images sized between 500x500 and 599x599 pixels: {num_under600}')
        print(f'Number of square images larger than or equal to 600x600 pixels: {above600}')