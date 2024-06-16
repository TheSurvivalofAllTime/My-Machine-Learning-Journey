########################################################################################################
########################################################################################################
#Author : Nikolin Prenga

# Created on 17 Jun, 2024

# This module contains various metrics, including precision, recall, sepcificity, intersection over union (IoU),
# Dice Coefficient specificity which will be imported for monitoring training and validation metrics during the model training process.

########################################################################################################
########################################################################################################




import warnings
import numpy as np 
import tensorflow
from keras import backend as K
warnings.simplefilter('ignore')
import tensorflow as tf
from tensorflow.keras import backend as K




###############################################################################################################################################################
# Sørensen-Dice Index
# Dice Similarity Coefficient (DSC)
def Dice_Coef(y_true, y_pred):
    # if len(y_true.shape) != 4 or len(y_pred.shape) != 4:
    #     raise ValueError("y_true and y_pred must be 4D tensors")
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    

    class_num = 1#y_true.shape[-1]
    epsilon = 10**(-9)
    total_dice_score = 0  # Initialize total_loss before the loop
    for iter in range(class_num):

        y_true_iter = K.flatten(y_true[:,:,:,iter])
        y_pred_iter = K.flatten(y_pred[:,:,:,iter])

        # Find the intersection for each class
        intersection = K.sum(y_true_iter * y_pred_iter)
        # Compute the dice score for each class
        dice_score = ((2. * intersection ) / (K.sum(y_true_iter) + K.sum(y_pred_iter) + epsilon))

        # Add the dice score for each class
        total_dice_score += dice_score  

    # Average the total_loss over the number of classes
    total_dice_score /= class_num  
    return total_dice_score

###########################################################################################################################
# Macro Precision / Average Precision Across Classes
def multi_class_precision(y_true, y_pred):

    epsilon = 10**(-9)

    class_num = 1#y_true.shape[-1]
    total_precision = 0
    for iter in range(class_num):
        y_true_iter = K.flatten(y_true[:,:,:,iter])
        y_pred_iter = K.flatten(y_pred[:,:,:,iter])
        # Counts all true positives since if an element in y_true is 1 and in y_pred is 0,
        # this single multiplication will be zero, and adding 0 won't increase the number of true positives.
        # Calculate true positives (y_true is 1 and y_pred is 1)
        true_positives = K.sum(K.round(K.clip(y_true_iter * y_pred_iter, 0, 1)))
        # K.clip(x, x_min, x_max) 
        # Here, we count the number of predicted positives that are represented by 1.
        # By summing these ones, you get the total count of predicted positives.

        predicted_positives = K.sum(K.round(K.clip(y_pred_iter, 0, 1)))
        # Calculate precision, add epsilon to avoid division by zero
        precision_score = true_positives / (predicted_positives + epsilon)
        # Add precision for each class
        total_precision += precision_score
    #Average the precision for all classes
    total_precision /= class_num
    return total_precision

###########################################################################################################################
#Macro Recall / Average Precision Across Classes
def multi_class_recall(y_true, y_pred):

    epsilon = 10**(-12)
    class_num = int(1)#y_true.shape[-1]
    total_recall = 0  # Initialize total_recall before the loop
    for iter in range(class_num):
        y_true_iter = K.flatten(y_true[:,:,:,iter])
        y_pred_iter = K.flatten(y_pred[:,:,:,iter])
        true_positives = K.sum(K.round(K.clip(y_true_iter * y_pred_iter, 0, 1)))
        # 
        actual_positives = K.sum(K.round(K.clip(y_true_iter, 0, 1)))
        # Calculate the recall score for each class
        recall_score = true_positives  / (actual_positives + epsilon)
        total_recall += recall_score  # Accumulate recall_score
    total_recall /= class_num  # Average the total_recall over the number of classes
    return total_recall

###########################################################################################################################
# Macro Specificity. / Average Specificity Across Classes

def multi_class_specificity(y_true, y_pred):
 
    num_classes =  1#y_true.shape[-1]
    epsilon = 10**(-12)
    total_specificity =0

    for iter in range(num_classes):

        y_true_iter = K.flatten(y_true[: , : , : , iter])
        y_pred_iter = K.flatten(y_pred[: , : , :, iter])
        true_negatives = K.sum(K.round(K.clip((1-y_true_iter)*(1-y_pred_iter), 0, 1  )))
        all_actual_negatives = K.sum(K.round(K.clip(1-y_true_iter, 0, 1  )))

        local_specificity = true_negatives/(all_actual_negatives+ epsilon)
        total_specificity +=local_specificity

    total_specificity = total_specificity/num_classes
    return total_specificity

########################################################################################################################
# Mean IoU (Intersection over Union):
import tensorflow as tf

def mean_iou_two(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.int32)
    y_true = tf.cast(y_true, tf.int32)
    
    intersection = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    union = tf.reduce_sum(tf.cast(y_true + y_pred, tf.float32)) - intersection
    
    iou = (intersection + 1e-10) / (union + 1e-10)
    return iou

# Custom Keras metric
def custom_mean_ioutow(y_true, y_pred):
    return mean_iou_two(y_true, y_pred)



def f1_score(y_true, y_pred):
    # y_true = tf.cast(K.flatten(y_true), 'int32')
    # y_pred = tf.cast(K.argmax(y_pred, axis=-1), 'int32')
    # y_pred = K.flatten(y_pred)
    
    # tp = K.sum(K.cast(y_true * y_pred, 'float32'))
    # fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'))
    # fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'))

    # precision = tp / (tp + fp + K.epsilon())
    # recall = tp / (tp + fn + K.epsilon())
    precision = multi_class_precision(y_true, y_pred)
    recall = multi_class_recall(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

