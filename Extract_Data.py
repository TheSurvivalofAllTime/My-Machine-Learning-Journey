import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
import numpy as np
import shutil
from git import Repo
import cv2
import plotly.graph_objects as go
from collections import Counter


######################################################################################################################
#######################################################################################################################
# Labeling Tumor Types
######################################################################################################################
#######################################################################################################################

class_names = {
    0: "No Tumor",
    1: "Glioma Tumor",
    2: "Meningioma tumor",
    3: "Pituitary tumor"
}




######################################################################################################################
#######################################################################################################################

# Obtaining Labels: Developing an Image Loading and Labeling Function for Training and Testing Datasets

######################################################################################################################
#######################################################################################################################

# Define label mappings based on the prefixes
label_mapping = {
    'g' : 1, # glioma_tumor
    'm' : 2, # meningioma_tumor
    'n' : 0, # no_tumor
    'p' : 3 # pituitary_tumor
}

# Function to extract the labels from the file images
def extract_label(fileimage):
    # Extract the first letter from the filename as pre
    prefix = fileimage[0]

    label_text = {
        "g": "glioma_tumor",  
        "m": "meningioma_tumor",  
        "n": "no_tumor",  
        "p": "pituitary_tumor"  
    }.get(prefix, "unknown")  # Get the label corresponding to the prefix, defaulting to "unknown" if prefix is not found

    # Get the corresponding label from the mapping
    label_numeric = label_mapping.get(prefix, -1)
    # the -1 is a default value returned by the get() method
    # if the prefix is not found in the label_mapping dictionary. 
    #It serves as a fallback value in case the prefix
    #  is not present in the dictionary.

    return label_text, label_numeric


def load_images_and_label_them(directory):
  #Add images at this object
    images =[]
    image_names = []
    label_numerics = []

    for imagename in os.listdir(directory):
        # Construct the full path to the image file
        image_path = os.path.join(directory, imagename)

        label_text, label_numeric = extract_label(imagename)
        
        # Check if the numeric label is valid (-1 indicates invalid label)
        
        if label_numeric !=-1:
          #Proceed with further processing only if the label is valid

          #load the images 
          image = cv2.imread(image_path)

          if image is not None:
              #Resize the images to 224 x 224 to a uniform shape
              image = cv2.resize(image, (220, 220))

              # Append the images, the numeric labels and the image names

              images.append(image)
              image_names.append(label_text)
              label_numerics.append(label_numeric)

    return np.array(images), image_names, label_numerics


######################################################################################################################
#######################################################################################################################
# Creating Python Function for Shuffling Data
######################################################################################################################
#######################################################################################################################

def load_data_and_shuffle(directory):
    np.random.seed(42)
    # Load images, filenames, and numeric labels from the specified directory
    images_data, filename, labels_numeric = load_images_and_label_them(directory)
    # Get the number of samples
    num_samples = len(images_data)
    # Generate shuffled indices
    shuffled_test_indices = np.random.permutation(num_samples)

    # Shuffle images, labels, and filenames using the shuffled indices
    images_data_shuffled = images_data[shuffled_test_indices]
    labels_numeric_shuffled = np.array(labels_numeric)[shuffled_test_indices]
    filename_shuffled = np.array(filename)[shuffled_test_indices]

    return images_data_shuffled, labels_numeric_shuffled, filename_shuffled


######################################################################################################################
#######################################################################################################################

## Obtaining the Final Shuffled Training and Testing Dataset with its Labels for Machine Learning Models

######################################################################################################################
#######################################################################################################################

# Define the directories containing the testing and training datasets
testing_dir = "Make_one_Testingdataset"
training_dir = "Make_one_Training_dataset"  # Or the directory where you have stored the training images
# Load and shuffle testing data
X_test_shuffled, Y_test_shuffled, Labels_test_shuffled = load_data_and_shuffle(testing_dir)

# Load and shuffle training data
X_train_shuffled, Y_train_shuffled, Labels_train_shuffled = load_data_and_shuffle(training_dir)


