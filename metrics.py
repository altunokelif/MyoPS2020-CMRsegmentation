# -*- coding: utf-8 -*-

import tensorflow.keras.backend as K

def multi_class_dice(y_true, y_pred, smooth=1e-5):    
    n_dims = len(y_pred.shape) - 2 
    axes = [axis for axis in range(n_dims + 1)] 
    intersect = K.sum(y_true * y_pred, axis=axes)
    numerator = 2 * intersect + smooth
    denominator = K.sum(y_true, axis = axes) + K.sum(y_pred, axis = axes) + smooth  
    return K.mean(numerator / denominator)

def multiclass_dice_loss(y_true, y_pred, smooth=1e-5):
    return 1-multi_class_dice(y_true, y_pred, smooth=1e-5)

#Defining dice coefficient for each class 

def dice_background(y_true, y_pred):  
  smooth=0
  y_true_f = K.flatten(y_true[:,:,:,0])
  y_pred_f = K.flatten(y_pred[:,:,:,0])
  intersection = K.sum(y_true_f * y_pred_f)
  dice=(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
  return dice

def dice_LVbloodpool(y_true, y_pred):
  smooth=0
  y_true_f = K.flatten(y_true[:,:,:,1])
  y_pred_f = K.flatten(y_pred[:,:,:,1])
  intersection = K.sum(y_true_f * y_pred_f)
  dice=(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
  return dice

def dice_RVbloodpool(y_true, y_pred):   
  smooth=0
  y_true_f = K.flatten(y_true[:,:,:,2])
  y_pred_f = K.flatten(y_pred[:,:,:,2])
  intersection = K.sum(y_true_f * y_pred_f)
  dice=(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
  return dice

def dice_LVmyo(y_true, y_pred):  
  smooth=0
  y_true_f = K.flatten(y_true[:,:,:,3])
  y_pred_f = K.flatten(y_pred[:,:,:,3])
  intersection = K.sum(y_true_f * y_pred_f)
  dice=(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
  return dice


def dice_LVmyoedema(y_true, y_pred): 
  smooth=0
  y_true_f = K.flatten(y_true[:,:,:,4])
  y_pred_f = K.flatten(y_pred[:,:,:,4])
  intersection = K.sum(y_true_f * y_pred_f)
  dice=(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
  return dice

def dice_LVmyoscars(y_true, y_pred):      
  smooth=0
  y_true_f = K.flatten(y_true[:,:,:,5])
  y_pred_f = K.flatten(y_pred[:,:,:,5])
  intersection = K.sum(y_true_f * y_pred_f)
  dice=(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
  return dice