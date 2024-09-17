import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd




# class YoloActivation(tf.keras.layers.Layer):
#     def call(self, inputs):
#         # Apply softmax to the class probabilities
#         class_probs = tf.nn.softmax(inputs[..., 0:3], axis=-1)
        
#         # Apply sigmoid to the bounding boxes (x, y, w, h) and confidence scores
#         boxes_conf = tf.sigmoid(inputs[..., 3:13])
        
#         # Concatenate the processed outputs
#         return tf.concat([class_probs, boxes_conf], axis=-1)





class YoloActivation(tf.keras.layers.Layer):
    def call(self, inputs):
        num_classes = 3  # Replace with the actual number of classes
        B = 2  # Number of bounding boxes
        
        # Softmax for class probabilities
        class_probs = tf.nn.softmax(inputs[..., :num_classes], axis=-1)
        
        # Sigmoid for box coordinates (x, y, w, h) and confidence scores
        box_confidence = tf.sigmoid(inputs[..., num_classes:])
        
        # Combine the processed outputs
        return tf.concat([class_probs, box_confidence], axis=-1)


def YOLOv1_Nikolin(input_shape, num_classes,drop_rate, B =2, S=7):
    
    input  = tf.keras.Input(shape=input_shape)
    
    # First layer with Batch Normalization
    Layer1 = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(input)
    Layer1 = layers.BatchNormalization()(Layer1)
    Layer1 = layers.LeakyReLU(alpha=0.1)(Layer1)

    #########################################################################################################################################
    # First Pooling Layer
    MaxPool1 = layers.MaxPool2D(pool_size=(2, 2), strides=2)(Layer1)

    #########################################################################################################################################
    # Second Layer with Batch Normalization
    Layer2 = layers.Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='same')(MaxPool1)
    Layer2 = layers.BatchNormalization()(Layer2)
    Layer2 = layers.LeakyReLU(alpha=0.1)(Layer2)

    #########################################################################################################################################
    #Second Pooling Layer
    MaxPool2 = layers.MaxPool2D(pool_size= (2,2), strides=2)(Layer2)

    #########################################################################################################################################
    #Third  Layer
    Layer3 = layers.Conv2D(filters=128, kernel_size =(1,1), strides =1, padding='same')(MaxPool2)
    Layer3 = layers.BatchNormalization()(Layer3)
    Layer3 = layers.LeakyReLU(alpha=0.1)(Layer3)

    #Fourth  Layer
    Layer4 = layers.Conv2D(filters=256, kernel_size =(3,3), strides =1, padding='same')(Layer3)
    Layer4 = layers.BatchNormalization()(Layer4)
    Layer4 = layers.LeakyReLU(alpha=0.1)(Layer4)

    #Fifth  Layer
    Layer5 = layers.Conv2D(filters=256, kernel_size =(1,1), strides =1, padding='same')(Layer4)
    Layer5 = layers.BatchNormalization()(Layer5)
    Layer5 = layers.LeakyReLU(alpha=0.1)(Layer5)

    #Sixth  Layer
    Layer6 = layers.Conv2D(filters=512, kernel_size =(3,3), strides =1, padding='same')(Layer5)
    Layer6 = layers.BatchNormalization()(Layer6)
    Layer6 = layers.LeakyReLU(alpha=0.1)(Layer6)

    #Third Pooling Layer
    MaxPool3 = layers.MaxPool2D(pool_size= (2,2), strides=2)(Layer6)

    #Seventh  Layer 
    Layer7 = layers.Conv2D(filters=256, kernel_size =(1,1), strides =1, padding='same')(MaxPool3)
    Layer7 = layers.BatchNormalization()(Layer7)
    Layer7 = layers.LeakyReLU(alpha=0.1)(Layer7)

    #Eighth  Layer
    Layer8 = layers.Conv2D(filters=512, kernel_size =(3,3), strides =1, padding='same')(Layer7)
    Layer8 = layers.BatchNormalization()(Layer8)
    Layer8 = layers.LeakyReLU(alpha=0.1)(Layer8)

    #Ninth  Layer 
    Layer9 = layers.Conv2D(filters=256, kernel_size =(1,1), strides =1, padding='same')(Layer8)
    Layer9 = layers.BatchNormalization()(Layer9)
    Layer9 = layers.LeakyReLU(alpha=0.1)(Layer9)

    Layer10 = layers.Conv2D(filters=512, kernel_size =(3,3), strides =1, padding='same')(Layer9)
    Layer10 = layers.BatchNormalization()(Layer10)
    Layer10 = layers.LeakyReLU(alpha=0.1)(Layer10)

    #11th  Layer 
    Layer11 = layers.Conv2D(filters=256, kernel_size =(1,1), strides =1, padding='same')(Layer10)
    Layer11 = layers.BatchNormalization()(Layer11)
    Layer11 = layers.LeakyReLU(alpha=0.1)(Layer11)
    
    #12th  Layer
    Layer12 = layers.Conv2D(filters=512, kernel_size =(3,3), strides =1, padding='same')(Layer11)
    Layer12 = layers.BatchNormalization()(Layer12)
    Layer12 = layers.LeakyReLU(alpha=0.1)(Layer12)

    #13th  Layer 
    Layer13 = layers.Conv2D(filters=256, kernel_size =(1,1), strides =1, padding='same')(Layer12)
    Layer13 = layers.BatchNormalization()(Layer13)
    Layer13 = layers.LeakyReLU(alpha=0.1)(Layer13)

    # #14th  Layer
    Layer14 = layers.Conv2D(filters=512, kernel_size =(3,3), strides =1, padding='same')(Layer13)
    Layer14 = layers.BatchNormalization()(Layer14)
    Layer14 = layers.LeakyReLU(alpha=0.1)(Layer14)

    #15th  Layer
    Layer15 = layers.Conv2D(filters=512, kernel_size =(1,1), strides =1, padding='same')(Layer14)
    Layer15 = layers.BatchNormalization()(Layer15)
    Layer15 = layers.LeakyReLU(alpha=0.1)(Layer15)

    #16th  Layer
    Layer16 = layers.Conv2D(filters=1024, kernel_size =(3,3), strides =1, padding='same')(Layer15)
    Layer16 = layers.BatchNormalization()(Layer16)
    Layer16 = layers.LeakyReLU(alpha=0.1)(Layer16)

    #Fourth Pooling Layer
    MaxPool4 = layers.MaxPool2D(pool_size= (2,2), strides=2)(Layer16)

    #17th  Layer
    Layer17 = layers.Conv2D(filters=512, kernel_size =(1,1), strides =1, padding='same')(MaxPool4)
    Layer17 = layers.BatchNormalization()(Layer17)
    Layer17 = layers.LeakyReLU(alpha=0.1)(Layer17)

    # 18th  Layer
    Layer18 = layers.Conv2D(filters=1024, kernel_size =(3,3), strides =1, padding='same')(Layer17)
    Layer18 = layers.BatchNormalization()(Layer18)
    Layer18 = layers.LeakyReLU(alpha=0.1)(Layer18)
    
    #19th  Layer
    Layer19 = layers.Conv2D(filters=512, kernel_size =(1,1), strides =1, padding='same')(Layer18)
    Layer19 = layers.BatchNormalization()(Layer19)
    Layer19 = layers.LeakyReLU(alpha=0.1)(Layer19)

    #20th  Layer
    Layer20 = layers.Conv2D(filters=1024, kernel_size =(3,3), strides =1, padding='same')(Layer19)
    Layer20 = layers.BatchNormalization()(Layer20)
    Layer20 = layers.LeakyReLU(alpha=0.1)(Layer20)

    #21th  Layer
    Layer21 = layers.Conv2D(filters=1024, kernel_size =(3,3), strides =1, padding='same')(Layer20)
    Layer21 = layers.BatchNormalization()(Layer21)
    Layer21 = layers.LeakyReLU(alpha=0.1)(Layer21)

    #22th  Layer
    Layer22 = layers.Conv2D(filters=1024, kernel_size =(3,3), strides =2, padding='same')(Layer21)
    Layer22 = layers.BatchNormalization()(Layer22)
    Layer22 = layers.LeakyReLU(alpha=0.1)(Layer22)

    #23th  Layer
    Layer23 = layers.Conv2D(filters=1024, kernel_size =(3,3), strides =1, padding='same')(Layer22)
    Layer23 = layers.BatchNormalization()(Layer23)
    Layer23 = layers.LeakyReLU(alpha=0.1)(Layer23)

    #24th  Layer
    Layer24 = layers.Conv2D(filters=1024, kernel_size =(3,3), strides =1, padding='same')(Layer23)
    Layer24 = layers.BatchNormalization()(Layer24)
    Layer24 = layers.LeakyReLU(alpha=0.1)(Layer24)
    
    #Flattening the layer 24
    Flattenimage = tf.keras.layers.Flatten()(Layer24)

    # Apply Dropout after flattening
    Dropout1 = tf.keras.layers.Dropout(rate=drop_rate)(Flattenimage)

    # First Fully Connected Layer
    FC1 = tf.keras.layers.Dense(units=4096, activation='relu')(Dropout1)

    # Apply Dropout after the first fully connected layer
    Dropout2 = tf.keras.layers.Dropout(rate=drop_rate)(FC1)

    # Second Fully Connected Layer
    FC2 = tf.keras.layers.Dense(units=4096, activation='relu')(Dropout2)

    # Output Layer: Adjust units according to the YOLO output format
    output = tf.keras.layers.Dense(units=S * S * (B * 5 + num_classes), activation='linear')(FC2)

    # Reshape to match the YOLO output format
    output = tf.keras.layers.Reshape((S, S, B * 5 + num_classes))(output)

    # Define the Model
    model = tf.keras.Model(inputs=input, outputs=output)

    # Optionally, print the model summary
    #model.summary()
    return model
    #######################################################################################################################
    # # Fully Connected Layers
    # FC1 = layers.Dense(4096, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(Dropout1)
    # FC1 = layers.Dropout(drop_rate)(FC1)
    
    # # Second fully connected layer (FC2)
    # FC2 = layers.Dense(4096, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(FC1)
    # FC2 = layers.Dropout(drop_rate)(FC2)
    
    # output_units = S * S * (B * 5 + num_classes)
    # output = layers.Dense(output_units)(FC2)  # No activation here
    # #print('here output :', output.shape)

    # output = YoloActivation()(output)  # Apply custom activation
    # #print(output.shape, '\n jjjj ___________________ \n')
    # #print(output.shape)
 
    
    # output = tf.reshape(output, (-1, S, S, B * 5 + num_classes))  # Reshape to the desired output shape
    # #output = tf.reshape(output, (S, S, B * 5 + num_classes))  # Reshape to the desired output shape
    
    # model = tf.keras.Model(inputs=input, outputs=output)
    
    # return model










   

    

