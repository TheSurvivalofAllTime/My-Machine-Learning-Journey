
########################################################################################################
########################################################################################################
#Author : Nikolin Prenga

# Created on 17 Jun, 2024

# This module serves as the foundation for creating the data loader.
# It is responsible for resizing images, generating masks, and preparing the final dataset
# for training and testing the models, ensuring that the data is properly formatted and ready for use.

########################################################################################################
########################################################################################################


from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def resolve_overlaps(mask_start, mask_single):
    """Resolve overlaps by combining masks using np.maximum."""
    return np.maximum(mask_start, mask_single)

def make_dataset(image_dir, label_dir, unseen_image_dir, im_height, im_width):

    images_indices = sorted(os.listdir(image_dir))
    mask_folders = sorted(os.listdir(label_dir))
    unseen_images_indices = sorted(os.listdir(unseen_image_dir))

    images = []
    masks = []
    unseen_images = []

    for test_image_file in unseen_images_indices:
        full_path_image = os.path.join(unseen_image_dir, test_image_file)
        image_unseen = cv2.imread(full_path_image, cv2.IMREAD_COLOR)
        if image_unseen is None:
            print(f'Image not found in unseen dataset: {full_path_image}')
            continue
        resized_unseen_image = cv2.resize(image_unseen, (im_height, im_width))
        unseen_images.append(resized_unseen_image)

    for image_file in images_indices:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Image not found: {image_path}")
            continue
        image_resized = cv2.resize(image, (im_height, im_width))
        images.append(image_resized)

    for mask_folder in mask_folders:
        mask_path = os.path.join(label_dir, mask_folder)
        mask_files = sorted(os.listdir(mask_path))

        mask_start = np.zeros((im_height, im_width, 1), dtype=np.uint8)
        valid_mask_found = False

        for mask_file in mask_files:
            path_single_mask = os.path.join(mask_path, mask_file)
            mask_single_image = cv2.imread(path_single_mask, cv2.IMREAD_GRAYSCALE)

            if mask_single_image is None:
                print(f"Mask not found: {path_single_mask}")
                continue   

            mask_single = cv2.resize(mask_single_image, (im_height, im_width))
            mask_single = np.expand_dims(mask_single, axis=-1)
            
            # Resolve overlaps
            mask_start = resolve_overlaps(mask_start, mask_single)
            valid_mask_found = True
        if valid_mask_found:
            masks.append(mask_start)
        else:
            print(f"No valid masks found in folder: {mask_folder}")

    images_all = np.array(images)
    masks_all = np.array(masks)
    unseen_images = np.array(unseen_images)
    
    if images_all.shape[0] != masks_all.shape[0]:
        print(f"Number of images ({images_all.shape[0]}) and masks ({masks_all.shape[0]}) do not match.")
    else:
        print("Number of images and masks match.")
    return images_all / 255.0, masks_all, unseen_images / 255.0





unseen_image_dir ='UnseenData'

image_dir = 'Data/Image_train'

label_dir = 'Data/Label_train'

All_imagage_train, All_masks_train, Unseen_data_images = make_dataset(image_dir, label_dir, unseen_image_dir, 128 ,128)

#All_imagage_train_y, All_masks_train_y, Unseen_data_images_y = make_dataset_solve(image_dir, label_dir, unseen_image_dir,  128 ,128)

#####################################################################################################################################################
#####################################################################################################################################################




Data_X, X_test, Data_y, y_test = train_test_split(All_imagage_train, All_masks_train, test_size=0.08, random_state=42)
print('\nShape of Data_X:', Data_X.shape)
print('Shape of X_test:', X_test.shape)
print('Shape of Data_y:', Data_y.shape)
print('Shape of y_test:', y_test.shape)



X_train, X_val , y_train, y_val = X_train, X_test, y_train, y_test = train_test_split(Data_X, Data_y, test_size=0.13)
print('\nShape of X_train:', X_train.shape)
print('Shape of X_val:', X_val.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of y_val:', y_val.shape)


# # Ensure the masks are binary
# y_train = (y_train > 0).astype(np.uint8)
# y_val = (y_val > 0).astype(np.uint8)
# y_test = (y_test > 0).astype(np.uint8)


#####################################################################################################################################################
#####################################################################################################################################################

# Data_X_gen, X_test_gen, Data_y_gen, y_test_gen = train_test_split(All_imagage_train, All_masks_train, test_size=0.08)
# X_train_gen, X_val_gen , y_train_gen, y_val_gen  = train_test_split(Data_X_gen, Data_y_gen, test_size=0.13)
# #print('.....  ', X_test_gen.shape)

# data_gen_args = dict(
#     rotation_range=90.,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True,
#     fill_mode='reflect'
# )

# image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(**data_gen_args)

# # Provide the same seed and keyword arguments to the fit and flow methods
# # Objects represented with ones's and background represented by 0s
# y_train_gen = (y_train_gen>0).astype(np.int8)
# y_test_gen = (y_test_gen>0).astype(np.int8)
# y_val_gen = (y_val_gen>0).astype(np.int8)

# image_datagen.fit(X_train_gen, augment=True)
# mask_datagen.fit(y_train_gen, augment=True)

# image_generator = image_datagen.flow(X_train_gen, batch_size=16)
# mask_generator = mask_datagen.flow(y_train_gen, batch_size=16)




# # Combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)


# #####################################################################################################################################################
# #####################################################################################################################################################

# X_train_all , X_val_all, Y_train_all, Y_val_all = train_test_split(All_imagage_train, All_masks_train, test_size=0.2 )

# image_datagen_all = ImageDataGenerator(**data_gen_args)
# mask_datagen_all = ImageDataGenerator(**data_gen_args)
# # Objects represented with ones's and background represented by 0s
# Y_train_all_modified =(Y_train_all>0).astype(np.int8)

# # Provide the same seed and keyword arguments to the fit and flow methods
# image_datagen_all.fit(X_train_all, augment=True)
# mask_datagen_all.fit(Y_train_all_modified, augment=True)  # Fixed to use mask_datagen_all

# plt.subplot(1,3,1)
# plt.imshow(Y_train_all_modified[2])

# plt.subplot(1,3,2)
# plt.imshow(X_train_all[2])
# plt.subplot(1,3,3)
# plt.imshow(Y_train_all[2])
# plt.show()




# image_generator_all = image_datagen_all.flow(X_train_all, batch_size=16)
# mask_generator_all = mask_datagen_all.flow(Y_train_all_modified, batch_size=16)

# # Combine generators into one which yields image and masks
# train_generator_all = zip(image_generator_all, mask_generator_all)




# def make_dataset(image_dir, label_dir, unseen_image_dir, im_height, im_width):

#     images_indices = sorted(os.listdir(image_dir))
#     mask_folders = sorted(os.listdir(label_dir))

#     unseen_images_indices = sorted(os.listdir(unseen_image_dir))

#     images = []
#     masks = []

#     unseen_images = []

#     for  test_image_file in unseen_images_indices:
#         full_path_image = os.path.join(unseen_image_dir , test_image_file)
#         image_unseen = cv2.imread(full_path_image, cv2.IMREAD_COLOR)

#         if image_unseen is None:
#             print(f'Image not found in unseen dataset: {full_path_image}')
#             continue
#         resized_unseen_image = cv2.resize(image_unseen, (im_height,im_width ))

#         unseen_images.append(resized_unseen_image)




    

#     for iter, image_file in enumerate(images_indices):
#         image_path = os.path.join(image_dir, image_file)
#         image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         if image is None:
#             print(f"Image not found: {image_path}")
#             continue
#         image_resized = cv2.resize(image, (im_height,im_width ))
#         images.append(image_resized)

#     for iter_mask, mask_folder in enumerate(mask_folders):
#         mask_path = os.path.join(label_dir,mask_folder )
#         mask_files = sorted(os.listdir(mask_path))

#         # Initialize the mask_start as an empty array
#         mask_start = np.zeros((im_height, im_width, 1), dtype=np.uint8)

#         valid_mask_found = False

#         for mask_file in mask_files:

#             path_single_mask = os.path.join(mask_path ,mask_file)

#             mask_single_image = cv2.imread(path_single_mask, cv2.IMREAD_GRAYSCALE)# IMREAD_GRAYSCALE

#             if mask_single_image is None:
#                 print(f"Mask not found: {path_single_mask}")
#                 continue

#             mask_single = cv2.resize(mask_single_image, (im_height, im_width))
#             mask_single = np.expand_dims(mask_single, axis=-1)

#             # Check for overlap: if any overlapping pixels exist, raise an error
#             # if np.any((mask_start > 0) & (mask_single > 0)):
#             #     raise ValueError(f"Overlapping masks found in folder: {mask_folder}")
            
#             mask_start = np.maximum(mask_start, mask_single) 
#             valid_mask_found = True
    
#         if valid_mask_found:
#                 masks.append(mask_start)
#         else:
#             print(f"No valid masks found in folder: {mask_folder}")

#     images_all = np.array(images)
#     masks_all = np.array(masks)
#     unseen_images = np.array(unseen_images)
#     if images_all.shape[0] != masks_all.shape[0]:
#         print(f"Number of images ({images_all.shape[0]}) and masks ({masks_all.shape[0]}) do not match.")
#     else:
#         print("Number of images and masks match.")
    

#     return images_all/255.0, masks_all, unseen_images/255.0