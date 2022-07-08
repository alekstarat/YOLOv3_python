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


def load_image_pixels(filename, shape):

    img = image.load_img(filename)
    width, height = img.size
    
    img = image.load_img(filename, target_size=shape)
    img = image.img_to_array(img)
    img = img.astype('float32')
    img /= 255.0
    img = np.expand_dims(img, 0)
    
    return img, width, height