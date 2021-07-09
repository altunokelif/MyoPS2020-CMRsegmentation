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


dropout=0.2
hn = 'he_normal'

def UNet(input_size = (256,256,3), num_classes=6):
  image_width = input_size[0]
  image_height = input_size[1]
  img_channels = input_size[2]

  inputs = tensorflow.keras.layers.Input((image_width, image_height, img_channels))
  s = tensorflow.keras.layers.Lambda(lambda x: x )(inputs)

  #Contraction path

  c1 = tensorflow.keras.layers.Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(s)
  c1 = tensorflow.keras.layers.Dropout(0.1)(c1)
  c1 = tensorflow.keras.layers.Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
  p1 = tensorflow.keras.layers.MaxPooling2D((2,2))(c1)
  p1 = tensorflow.keras.layers.MaxPooling2D((2,2), strides=(2,2))(c1)
    
  c2 = tensorflow.keras.layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
  c2 = tensorflow.keras.layers.Dropout(0.2)(c2)
  c2 = tensorflow.keras.layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
  p2 = tensorflow.keras.layers.MaxPooling2D((2,2), strides=(2,2))(c2)

  c3 = tensorflow.keras.layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
  c3 = tensorflow.keras.layers.Dropout(0.2)(c3)
  c3 = tensorflow.keras.layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
  p3 = tensorflow.keras.layers.MaxPooling2D((2,2),strides=(2,2))(c3)

  c4 = tensorflow.keras.layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
  c4 = tensorflow.keras.layers.Dropout(0.3)(c4)
  c4 = tensorflow.keras.layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
  p4 = tensorflow.keras.layers.MaxPooling2D((2,2),strides=(2,2))(c4)

  c5 = tensorflow.keras.layers.Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
  c5 = tensorflow.keras.layers.Dropout(0.1)(c5)
  c5 = tensorflow.keras.layers.Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)

  # Expansive path

  u6 = tensorflow.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding="same" )(c5)
  u6 = tensorflow.keras.layers.concatenate([u6,c4])
  c6 =  tensorflow.keras.layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
  c6 = tensorflow.keras.layers.Dropout(0.2)(c6)
  c6 =  tensorflow.keras.layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

  u7 = tensorflow.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding="same" )(c6)
  u7 = tensorflow.keras.layers.concatenate([u7,c3])
  c7 =  tensorflow.keras.layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
  c7 = tensorflow.keras.layers.Dropout(0.2)(c7)
  c7 = tensorflow.keras.layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)


  u8 = tensorflow.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding="same" )(c7)
  u8 = tensorflow.keras.layers.concatenate([u8,c2])
  c8 = tensorflow.keras.layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
  c8 = tensorflow.keras.layers.Dropout(0.1)(c8)
  c8 = tensorflow.keras.layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)


  u9 = tensorflow.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding="same" )(c8)
  u9 = tensorflow.keras.layers.concatenate([u9,c1], axis=3)
  c9 = tensorflow.keras.layers.Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
  c9 = tensorflow.keras.layers.Dropout(0.1)(c9)
  c9 = tensorflow.keras.layers.Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

  outputs = tensorflow.keras.layers.Conv2D(num_classes, (1,1), activation='sigmoid')(c9)

  model = tensorflow.keras.Model(inputs=[inputs], outputs=[outputs])
  
  return model
