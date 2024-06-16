########################################################################################################
########################################################################################################
#Author : Nikolin Prenga

# Created on 17 Jun, 2024

# This module implements U-Net and Fully Convolutional Networks (FCN) 
# to be used during training and testing. By importing these models, 
# we can streamline the code and avoid redundancy in the training and testing scripts.

########################################################################################################
########################################################################################################



import tensorflow
from keras import backend as K
import shutil 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.callbacks import CSVLogger
from tensorflow.keras import layers
from tensorflow.keras import layers, models




def My_Unet_2D(input_size, num_filter, Pool_size,  pool_stride, drop_rate, num_classes):
    input = tf.keras.layers.Input(shape = input_size)
    filter = num_filter

    #########################################################################################################
    # Downsampling: Encoder or the contraction path
    #########################################################################################################
    # This section of the U-Net architecture is responsible for reducing the spatial dimensions of the input
    # image while increasing the number of feature channels. It consists of multiple convolutional blocks
    # followed by max-pooling layers.



    #########################################################################################################
    # First Block: 2 convolutional layers followed by a max pooling layer and a dropout layer
    #########################################################################################################


    Block1_Conv1 = tf.keras.layers.Conv2D(filters = filter, kernel_size =3,
                                          padding='same',activation ='relu' )(input)
    Block1_Conv2 = tf.keras.layers.Conv2D(filters = filter, kernel_size =3,
                                           padding='same',activation ='relu' )(Block1_Conv1)

    Pooling_block1 = tf.keras.layers.MaxPooling2D( pool_size=Pool_size,
                                                      strides=pool_stride, padding="valid" )(Block1_Conv2)

    Block1_Dropout = tf.keras.layers.Dropout(rate =drop_rate)(Pooling_block1)

    #################################################################################################################
    # Second Block: 2 convolutional layers followed by a max pooling layer and a dropout layer
    # The number of filters doubles
    ##################################################################################################################

    filter = int(filter*2)

    Block2_Conv3 = tf.keras.layers.Conv2D(filters = filter, kernel_size =3,
                                           padding='same', activation ='relu')(Block1_Dropout)

    Block2_Conv4 = tf.keras.layers.Conv2D(filters = filter, kernel_size =3,
                                           padding='same',activation ='relu')(Block2_Conv3)

    Pooling_block2 = tf.keras.layers.MaxPool2D(pool_size= Pool_size,
                                                      strides=pool_stride, padding ='valid')(Block2_Conv4)

    Block2_Dropout = tf.keras.layers.Dropout(rate =drop_rate)(Pooling_block2)


    ####################################################################################################################
    # Third Block: Consists of 2 convolutional layers followed by a max pooling layer and a dropout layer.
    # The number of filters doubles in this block.
    ######################################################################################################################

    filter = int(filter*2)

    Block3_Conv5 = tf.keras.layers.Conv2D(filters = filter, kernel_size=3,
                                           padding ='same', activation='relu')(Block2_Dropout)

    Block3_Conv6 = tf.keras.layers.Conv2D(filters = filter, kernel_size=3,
                                           padding ='same', activation='relu')(Block3_Conv5)

    Pooling_block3 = tf.keras.layers.MaxPooling2D(pool_size =Pool_size,
                                                  strides = pool_stride, padding ='valid' )(Block3_Conv6)
    Block3_Dropout = tf.keras.layers.Dropout(rate =drop_rate)(Pooling_block3)


    ####################################################################################################################
    # Fourth Block: Consists of 2 convolutional layers followed by a max pooling layer and a dropout layer.
    # The number of filters doubles in this block.
    ######################################################################################################################

    filter = int(filter*2)
    Block4_Conv7 = tf.keras.layers.Conv2D(filters = filter, kernel_size=3,
                                           padding ='same', activation='relu')(Block3_Dropout)

    Block4_Conv8 = tf.keras.layers.Conv2D(filters = filter, kernel_size=3,
                                           padding ='same', activation='relu')(Block4_Conv7)
    
    #Pooling_block4 = tf.keras.layers.MaxPooling3D(pool_size=Pool_size, strides =pool_stride )(Block4_Conv8)
    Pooling_block4 = tf.keras.layers.MaxPooling2D(pool_size=Pool_size, strides=pool_stride, padding='valid')(Block4_Conv8)


    block4_Dropout= tf.keras.layers.Dropout(rate=drop_rate)(Pooling_block4)

    ####################################################################################################################
    # Bottle Neck: This block serves as the bottleneck of the U-Net architecture. It typically consists of two
    # convolutional layers followed by a dropout without any max pooling or dropout layers.
    ####################################################################################################################

    filter = int(filter*2)
    BottleNeck_Conv9 = tf.keras.layers.Conv2D(filters = filter, kernel_size=3,
                                           padding ='same', activation='relu')(block4_Dropout)

    BottleNeck_Conv10 = tf.keras.layers.Conv2D(filters = filter, kernel_size=3,
                                           padding ='same', activation='relu')(BottleNeck_Conv9)

    BottleNeck_Dropout = tf.keras.layers.Dropout(rate= drop_rate)(BottleNeck_Conv10)

    ##################################################################################################################
    #Decoder Block (Upsampling / Extraction Path)
    ##################################################################################################################
    ##################################################################################################################
    # Block 6: Upsamples the spatial dimensions, concatenates features from Block 4, and applies two convolutional layers
    # The number of filters is halved in this block.
    ##################################################################################################################

    filter = int(filter/2)

    Block_6_upsample = tf.keras.layers.Conv2DTranspose(filters=filter,
                                                       kernel_size =3, strides =2,
                                                       padding ='same', activation='relu')(BottleNeck_Dropout)
    Block6_merge = tf.keras.layers.concatenate([Block_6_upsample, Block4_Conv8 ])
    print('Block_6_upsample', '--', Block_6_upsample.shape)
    print('Block4_Conv8', '--', Block4_Conv8.shape)

    Block6_Dropout = tf.keras.layers.Dropout(rate=drop_rate)(Block6_merge)

    Block6_Conv11 =tf.keras.layers.Conv2D(filters=filter, kernel_size=3,
                                         strides=1, padding='same', activation='relu')(Block6_Dropout)
    Block6_Conv12 = tf.keras.layers.Conv2D(filters=filter, kernel_size=3,
                                         strides=1, padding='same', activation='relu')(Block6_Conv11)


    ##################################################################################################################
    # Block 7: Upsamples the spatial dimensions, concatenates features from Block 3, and applies two convolutional layers
    # The number of filters is halved in this block.
    ##################################################################################################################
    filter = int(filter/2)

    Block7_upsample = tf.keras.layers.Conv2DTranspose(filters= filter,
                                                      kernel_size= 3, strides=2,
                                                      padding='same', activation='relu')(Block6_Conv12)

    Block7_merge =tf.keras.layers.concatenate([Block7_upsample, Block3_Conv6 ])
    Block7_Dropout = tf.keras.layers.Dropout(rate=drop_rate)(Block7_merge)

    Block7_Conv13 = tf.keras.layers.Conv2D(filters=filter,  kernel_size=3,
                                         strides=1, padding='same', activation='relu')(Block7_Dropout)
    Block7_Conv14 = tf.keras.layers.Conv2D(filters=filter, kernel_size=3,
                                         strides=1, padding='same', activation='relu')(Block7_Conv13)

    ##################################################################################################################
    # Block 8: Upsamples the spatial dimensions, concatenates features from Block 2, and applies two convolutional layers
    # The number of filters is halved in this block.
    ##################################################################################################################
    filter = int(filter/2)

    Block8_upsample = tf.keras.layers.Conv2DTranspose(filters= filter,
                                                      kernel_size=3, strides=2,
                                                      padding='same', activation='relu')(Block7_Conv14)
    Block8_merge = tf.keras.layers.concatenate([Block8_upsample, Block2_Conv4])

    Block8_Dropout = tf.keras.layers.Dropout(rate= drop_rate)(Block8_merge)

    Block8_Conv15 = tf.keras.layers.Conv2D(filters=filter, kernel_size=3,
                                         strides=1, padding='same', activation='relu' )(Block8_Dropout)

    Block8_Conv16 = tf.keras.layers.Conv2D(filters=filter, kernel_size=3,
                                         strides=1, padding='same', activation='relu' )(Block8_Conv15)


    ####################################################################################################################
    # Block 9: Output Layer - Upsamples the spatial dimensions, concatenates features from Block 1,
    # and applies two convolutional layers The number of filters is halved in this block.
    # The number of filters is halved in this block.
    ####################################################################################################################
    filter = int(filter/2)

    Block9_upsample= tf.keras.layers.Conv2DTranspose(filters= filter,
                                                      kernel_size=3, strides=2,
                                                      padding='same', activation='relu')(Block8_Conv16)

    Block9_merge = tf.keras.layers.concatenate([Block9_upsample, Block1_Conv2])

    

    Block9_Dropout = tf.keras.layers.Dropout(rate=drop_rate)(Block9_merge)

    Block9_Conv17 = tf.keras.layers.Conv2D(filters=filter, kernel_size=3,
                                         strides=1, padding='same', activation='relu' )(Block9_Dropout)
    Block9_Conv18 = tf.keras.layers.Conv2D(filters=filter, kernel_size=3,
                                         strides=1, padding='same', activation='relu' )(Block9_Conv17)
    
    

    # Output lAYER
    if isinstance(num_classes, int) and num_classes >= 2:
        if num_classes ==2:
          output = tf.keras.layers.Conv2D(1, kernel_size=(1,1), padding='same', activation='sigmoid')(Block9_Conv18)
          print('Sigmoid used')

        elif num_classes >2:
            output = tf.keras.layers.Conv2D(num_classes, kernel_size=1, padding='same', activation='softmax')(Block9_Conv18)
            print('Softmax used')
    else:
        print('Number of classes must be an integer and greater than or equal to 2. Try again!')

    model = tf.keras.Model(inputs=input, outputs = output, name ='MyUnetArchitecture' )

    return model




###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################




def FullyConvolutional_NeuralNetWork(input_shape, filter_choice, drop_rate, num_classes):
    shape_input = tf.keras.Input(shape=input_shape)

    ##############################################################################################################################
    # Encoder: VGG-like architecture with doubling filters
    # Block 1
    x = layers.Conv2D(filter_choice, (3, 3), activation='relu', padding='same')(shape_input)
    x = layers.Conv2D(filter_choice, (3, 3), activation='relu', padding='same')(x)
    block_1_output = layers.MaxPooling2D((2, 2))(x)

    ##############################################################################################################################
    # Block 2
    filter_choice *= 2
    x = layers.Conv2D(filter_choice, (3, 3), activation='relu', padding='same')(block_1_output)
    x = layers.Conv2D(filter_choice, (3, 3), activation='relu', padding='same')(x)
    block_2_output = layers.MaxPooling2D((2, 2))(x)
    ##############################################################################################################################
    # Block 3

    filter_choice *= 2
    x = layers.Conv2D(filter_choice, (3, 3), activation='relu', padding='same')(block_2_output)
    x = layers.Conv2D(filter_choice, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(filter_choice, (3, 3), activation='relu', padding='same')(x)
    block_3_output = layers.MaxPooling2D((2, 2))(x)

    ##############################################################################################################################
    # Block 4
    filter_choice *= 2

    x = layers.Conv2D(filter_choice, (3, 3), activation='relu', padding='same')(block_3_output)
    x = layers.Conv2D(filter_choice, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(filter_choice, (3, 3), activation='relu', padding='same')(x)
    block_4_output = layers.MaxPooling2D((2, 2))(x)

    ##############################################################################################################################
    # Block 5
    filter_choice *= 2

    x = layers.Conv2D(filter_choice, (3, 3), activation='relu', padding='same')(block_4_output)
    x = layers.Conv2D(filter_choice, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(filter_choice, (3, 3), activation='relu', padding='same')(x)
    block_5_output = layers.MaxPooling2D((2, 2))(x)

    ##############################################################################################################################
    # Decoder with dropout
    x = layers.Conv2D(4096, (7, 7), activation='relu', padding='same')(block_5_output)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = layers.Dropout(drop_rate)(x)

    x = layers.Conv2D(num_classes, (1, 1), activation='linear', padding='valid')(x)

    ##############################################################################################################################
    # Upsample to block 4 size
    x = layers.Conv2DTranspose(num_classes, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.add([x, layers.Conv2D(num_classes, (1, 1), activation='linear', padding='same')(block_4_output)])

    ##############################################################################################################################
    # Upsample to block 3 size
    x = layers.Conv2DTranspose(num_classes, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.add([x, layers.Conv2D(num_classes, (1, 1), activation='linear', padding='same')(block_3_output)])
    # x = layers.add([x, block_3_output])

    ##############################################################################################################################
    # Upsample to input size
    x = layers.Conv2DTranspose(num_classes, (16, 16), strides=(8, 8), padding='same')(x)
    
    ##############################################################################################################################
    # Output layer

    if num_classes>2:
        output =  layers.Activation('softmax')(x)
    else:
        output =  layers.Activation('sigmoid')(x)

    model = tf.keras.Model(inputs=shape_input, outputs=output)
    return model




###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################






def My_Unet_3D(input_size, num_filter, Pool_size, pool_stride, drop_rate, num_classes):
    input = tf.keras.layers.Input(shape=input_size)
    filter = num_filter

    #########################################################################################################
    # Downsampling: Encoder or the contraction path
    #########################################################################################################

    # First Block
    Block1_Conv1 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, padding='same', activation='relu')(input)
    Block1_Conv2 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, padding='same', activation='relu')(Block1_Conv1)
    Pooling_block1 = tf.keras.layers.MaxPooling3D(pool_size=Pool_size, strides=pool_stride, padding="valid")(Block1_Conv2)
    Block1_Dropout = tf.keras.layers.Dropout(rate=drop_rate)(Pooling_block1)

    # Second Block
    filter = int(filter * 2)
    Block2_Conv3 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, padding='same', activation='relu')(Block1_Dropout)
    Block2_Conv4 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, padding='same', activation='relu')(Block2_Conv3)
    Pooling_block2 = tf.keras.layers.MaxPool3D(pool_size=Pool_size, strides=pool_stride, padding='valid')(Block2_Conv4)
    Block2_Dropout = tf.keras.layers.Dropout(rate=drop_rate)(Pooling_block2)

    # Third Block
    filter = int(filter * 2)
    Block3_Conv5 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, padding='same', activation='relu')(Block2_Dropout)
    Block3_Conv6 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, padding='same', activation='relu')(Block3_Conv5)
    Pooling_block3 = tf.keras.layers.MaxPooling3D(pool_size=Pool_size, strides=pool_stride, padding='valid')(Block3_Conv6)
    Block3_Dropout = tf.keras.layers.Dropout(rate=drop_rate)(Pooling_block3)

    # Fourth Block
    filter = int(filter * 2)
    Block4_Conv7 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, padding='same', activation='relu')(Block3_Dropout)
    Block4_Conv8 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, padding='same', activation='relu')(Block4_Conv7)
    Pooling_block4 = tf.keras.layers.MaxPooling3D(pool_size=Pool_size, strides=pool_stride, padding='valid')(Block4_Conv8)
    Block4_Dropout = tf.keras.layers.Dropout(rate=drop_rate)(Pooling_block4)

    # Bottleneck
    filter = int(filter * 2)
    BottleNeck_Conv9 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, padding='same', activation='relu')(Block4_Dropout)
    BottleNeck_Conv10 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, padding='same', activation='relu')(BottleNeck_Conv9)
    BottleNeck_Dropout = tf.keras.layers.Dropout(rate=drop_rate)(BottleNeck_Conv10)

    ##################################################################################################################
    # Decoder Block (Upsampling / Extraction Path)
    ##################################################################################################################

    # Block 6: Upsample and concatenate with Block 4
    filter = int(filter / 2)
    Block_6_upsample = tf.keras.layers.Conv3DTranspose(filters=filter, kernel_size=3, strides=2, padding='same', activation='relu')(BottleNeck_Dropout)
    Block6_merge = tf.keras.layers.concatenate([Block_6_upsample, Block4_Conv8])
    Block6_Dropout = tf.keras.layers.Dropout(rate=drop_rate)(Block6_merge)
    Block6_Conv11 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, strides=1, padding='same', activation='relu')(Block6_Dropout)
    Block6_Conv12 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, strides=1, padding='same', activation='relu')(Block6_Conv11)

    # Block 7: Upsample and concatenate with Block 3
    filter = int(filter / 2)
    Block7_upsample = tf.keras.layers.Conv3DTranspose(filters=filter, kernel_size=3, strides=2, padding='same', activation='relu')(Block6_Conv12)
    Block7_merge = tf.keras.layers.concatenate([Block7_upsample, Block3_Conv6])
    Block7_Dropout = tf.keras.layers.Dropout(rate=drop_rate)(Block7_merge)
    Block7_Conv13 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, strides=1, padding='same', activation='relu')(Block7_Dropout)
    Block7_Conv14 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, strides=1, padding='same', activation='relu')(Block7_Conv13)

    # Block 8: Upsample and concatenate with Block 2
    filter = int(filter / 2)
    Block8_upsample = tf.keras.layers.Conv3DTranspose(filters=filter, kernel_size=3, strides=2, padding='same', activation='relu')(Block7_Conv14)
    Block8_merge = tf.keras.layers.concatenate([Block8_upsample, Block2_Conv4])
    Block8_Dropout = tf.keras.layers.Dropout(rate=drop_rate)(Block8_merge)
    Block8_Conv15 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, strides=1, padding='same', activation='relu')(Block8_Dropout)
    Block8_Conv16 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, strides=1, padding='same', activation='relu')(Block8_Conv15)

    # Block 9: Upsample and concatenate with Block 1
    filter = int(filter / 2)
    Block9_upsample = tf.keras.layers.Conv3DTranspose(filters=filter, kernel_size=3, strides=2, padding='same', activation='relu')(Block8_Conv16)
    Block9_merge = tf.keras.layers.concatenate([Block9_upsample, Block1_Conv2])
    Block9_Dropout = tf.keras.layers.Dropout(rate=drop_rate)(Block9_merge)
    Block9_Conv17 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, strides=1, padding='same', activation='relu')(Block9_Dropout)
    Block9_Conv18 = tf.keras.layers.Conv3D(filters=filter, kernel_size=3, strides=1, padding='same', activation='relu')(Block9_Conv17)

    # Output Layer
    if num_classes == 2:
        output = tf.keras.layers.Conv3D(1, kernel_size=(1, 1, 1), padding='same', activation='sigmoid')(Block9_Conv18)
        print('Sigmoid used')
    elif num_classes > 2:
        output = tf.keras.layers.Conv3D(num_classes, kernel_size=(1, 1, 1), padding='same', activation='softmax')(Block9_Conv18)
        print('Softmax used')
    else:
        raise ValueError('Number of classes must be an integer and greater than or equal to 2. Try again!')

    model = tf.keras.Model(inputs=input, outputs=output, name='MyUnetArchitecture')

    return model
