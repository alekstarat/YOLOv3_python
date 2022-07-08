import struct
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Lambda
from keras.optimizers import Adam 
from keras.layers import add, concatenate
from keras.models import Model
from keras.models import load_model
from itertools import repeat
from keras.regularizers import l2 
from keras.preprocessing import image 
from keras.utils.vis_utils import plot_model 
import tensorflow as tf 
from matplotlib.patches import Rectangle
from glob import glob
import os
import gdown
from matplotlib import pyplot as plt


                                                                               # LAYERS #

size = 416

channels= 3
num_sub_anchors=3
num_classes=80
inputs = Input([size, size, channels])


def DBL(x, filters, kernel, strides=1, batch_norm=True, layer_idx=None): 
    
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=kernel,
              strides=strides, padding=padding,
              use_bias=not batch_norm, kernel_regularizer=l2(0.0005), name='conv_' + str(layer_idx))(x)
    if batch_norm:
        x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(layer_idx))(x)
        x = LeakyReLU(alpha=0.1,name='leake_' + str(layer_idx))(x)
    
    return x, layer_idx+1


def Res_unit(x, filters, layer_idx): 

    skip_connection = x
    x, layer_idx = DBL(x, filters // 2, kernel=1, layer_idx=layer_idx)
    x, layer_idx = DBL(x, filters, kernel=3, layer_idx=layer_idx)
    x = add([skip_connection , x], name='Add_'+str(layer_idx))

    return x, layer_idx+1


def ResBlock(x, filters, blocks, layer_idx): 

    x, layer_idx = DBL(x, filters, kernel=3, strides=2, layer_idx=layer_idx)
    
    for _ in repeat(None, blocks):
        x, layer_idx = Res_unit(x, filters, layer_idx=layer_idx)
    
    return x, layer_idx


def Detector(x_in, filters, layer_idx=None):

    if isinstance(x_in, list): 
        x, x_skip = x_in[0], x_in[1]
        x,layer_idx = DBL(x, filters, kernel=1, strides=1, layer_idx=layer_idx) 
        x = UpSampling2D(2, name = 'UpSampling_' + str(layer_idx))(x) 
        layer_idx+=1
        x =concatenate([x, x_skip], name = 'Concatenate_' + str(layer_idx)) 
        layer_idx+=1
        
        for i in range(2):
          x, layer_idx = DBL(x, filters, 1, layer_idx=layer_idx)      
          x, layer_idx = DBL(x, filters * 2, 3, layer_idx=layer_idx)  
        
        fork, layer_idx = DBL(x, filters, 1, layer_idx=layer_idx)            
    
    else: 
        x = x_in
        
        for i in range(2):
          x, layer_idx = DBL(x, filters, 1, layer_idx=layer_idx)      
          x, layer_idx = DBL(x, filters * 2, 3, layer_idx=layer_idx)  
        
        fork, layer_idx = DBL(x, filters, 1, layer_idx=layer_idx)      

    x,layer_idx = DBL(fork, filters=filters*2, kernel=3, strides=1, layer_idx=layer_idx)
 
    bboxes, layer_idx = DBL(x, filters=num_sub_anchors * (4 + 1 + num_classes), kernel=1, strides=1, batch_norm= False, layer_idx=layer_idx)       

    return bboxes, fork, layer_idx


def create_yolov3_model(inputs, num_sub_anchors, num_classes):
    
    layer_idx = 0 
    x, layer_idx = DBL(inputs, filters=32, kernel=3, layer_idx=layer_idx)       
    x, layer_idx = ResBlock(x, filters=64, blocks=1, layer_idx=layer_idx)            
    x, layer_idx = ResBlock(x, filters=128, blocks=2, layer_idx=layer_idx)           
    x, layer_idx = Route_1,_ = ResBlock(x, filters=256, blocks=8, layer_idx=layer_idx) 
    x, layer_idx = Route_2,_ = ResBlock(x, filters=512, blocks=8, layer_idx=layer_idx) 
    Route_3, layer_idx = ResBlock(x, filters=1024, blocks=4, layer_idx=layer_idx)          
    
    bbox_scale_1, fork_1, layer_idx = Detector(Route_3, filters=512, layer_idx=layer_idx) 

    layer_idx = 84
    bbox_scale_2, fork_2, layer_idx = Detector([fork_1, Route_2], filters=256, layer_idx=layer_idx) 
 
    layer_idx = 96
    bbox_scale_3, _, layer_idx = Detector([fork_2, Route_1], filters=128, layer_idx=layer_idx) 

    model = Model (inputs, [bbox_scale_1, bbox_scale_2, bbox_scale_3])

    return model


model = create_yolov3_model(inputs, num_sub_anchors, num_classes)
model.summary()