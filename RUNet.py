# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Cropping2D, Conv2D, UpSampling2D, concatenate, Input,add, Dropout, Permute, Conv2D, BatchNormalization, Activation,Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# Seeding 
seed = 2019
random.seed = seed
np.random.seed = seed
#tf.seed = seed
from IPython.display import HTML, display, clear_output, SVG

K.set_image_data_format('channels_last')  

def residual_unit(inputx, kernel_size, filters, stage):
    filters1, filters2, filters3 = filters
   
    a = Conv2D(filters1, (1, 1))(inputx)
    a = BatchNormalization(axis=3)(a)
    a = Activation('relu')(a)
    a = Conv2D(filters2, kernel_size, padding='same')(a)
    a = BatchNormalization(axis=3)(a)
    a = Activation('relu')(a)
    a = Conv2D(filters3, (1, 1))(a)
    a = BatchNormalization(axis=3)(a)
    a = layers.add([a, inputx])
    a = Activation('relu')(a)
    return a


def downsampling_block(inputx, kernel_size, filters, stage, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    a = Conv2D(filters1, (1, 1), strides=strides)(inputx)
    a = BatchNormalization(axis=3)(a)
    a = Activation('relu')(a)
    a = Conv2D(filters2, kernel_size, padding='same')(a)
    a = BatchNormalization(axis=3)(a)
    a = Activation('relu')(a)
    a = Conv2D(filters3, (1, 1))(a)
    a = BatchNormalization(axis=3)(a)
    shortcut = Conv2D(filters3, (1, 1), strides=strides)(inputx)
    shortcut = BatchNormalization(axis=3)(shortcut)
    a = layers.add([a, shortcut])
    a = Activation('relu')(a)
    return a


def upsampling_block(inputx, kernel_size, filters, stage, strides=(1, 1)):
    filters1, filters2, filters3 = filters
  
    a = UpSampling2D(size=(2, 2))(inputx)
    a = Conv2D(filters1, (1, 1), strides=strides)(a)
    a = BatchNormalization(axis=3)(a)
    a = Activation('relu')(a)
    a = Conv2D(filters2, kernel_size, padding='same')(a)
    a = BatchNormalization(axis=3)(a)
    a = Activation('relu')(a)
    a = Conv2D(filters3, (1, 1))(a)
    a = BatchNormalization(axis=3)(a)
    shortcut = UpSampling2D(size=(2, 2))(inputx)
    shortcut = Conv2D(filters3, (1, 1), strides=strides)(shortcut)
    shortcut = BatchNormalization(axis=3)(shortcut)
    a = layers.add([a, shortcut])
    a = Activation('relu')(a)
    return a


def RUNet(i=16, axis=3, classes=6, height=256, width=256):
    
    #Before making operations in the encoding path
    input = Input((height, width, 3))
    a = ZeroPadding2D((5, 5))(input)
    #a = ZeroPadding2D((4, 4))(input)
    a = Conv2D(i, (7, 7), strides=(2, 2))(a)
    a = BatchNormalization(axis=3)(a)
    a = Activation('relu')(a)
    a = MaxPooling2D((3, 3), strides=(2, 2))(a)
    
    #Encoding path
    a = downsampling_block(a, 3, [i, i, i*2], stage=2,  strides=(1, 1))
    a = residual_unit(a, 3, [i, i, i*2], stage=2 )
    a2 = residual_unit(a, 3, [i, i, i*2], stage=2)
    a = downsampling_block(a2, 3, [i*2, i*2, i*4], stage=3)
    a = residual_unit(a, 3, [i*2, i*2, i*4], stage=3)
    a3 = residual_unit(a, 3, [i*2, i*2, i*4], stage=3)
    a = downsampling_block(a3, 3, [i*4, i*4, i*8], stage=4)
    a = residual_unit(a, 3, [i*4, i*4, i*8], stage=4)
    a4 = residual_unit(a, 3, [i*4, i*4, i*8], stage=4)
    
    #Bridge
    a = downsampling_block(a4, 3, [i*8, i*8, i*16], stage=5)
    a = residual_unit(a, 3, [i*8, i*8, i*16], stage=5)
    a = residual_unit(a, 3, [i*8, i*8, i*16], stage=5)
    
    #Decoding path
    a = upsampling_block(a, 3, [i*16, i*8, i*8], stage=6)
    a = residual_unit(a, 3, [i*16, i*8, i*8], stage=6)
    a = residual_unit(a, 3, [i*16, i*8, i*8], stage=6)
    a = concatenate([a, a4], axis=3)
    a = upsampling_block(a, 3, [i*16, i*4, i*4], stage=7)
    a = residual_unit(a, 3, [i*16, i*4, i*4], stage=7)
    a = residual_unit(a, 3, [i*16, i*4, i*4], stage=7)
    a = concatenate([a, a3], axis=3)
    a = upsampling_block(a, 3, [i*8, i*2, i*2], stage=8)
    a = residual_unit(a, 3, [i*8, i*2, i*2], stage=8)
    a = residual_unit(a, 3, [i*8, i*2, i*2], stage=8)
    a = concatenate([a, a2], axis=3)
    
    #After last decoding unit
    a = upsampling_block(a, 3, [i*4, i, i], stage=10, strides=(1, 1))
    a = residual_unit(a, 3, [i*4, i, i], stage=10)
    a = residual_unit(a, 3, [i*4, i, i], stage=10)
    a = UpSampling2D(size=(2, 2))(a)
    
    #Last layer
    a = Conv2D(classes, (1, 1), padding='same', activation='sigmoid')(a)
    #a = Conv2D(classes, (3, 3), padding='same', activation='sigmoid')(a)
    model = Model(input, a, name='RUNet')
    
   
    return model