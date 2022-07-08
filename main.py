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
from model import *
from weights import *
from bboxes import *
from img_defs import *


class_threshold = 0.5 
weight_reader = WeightReader('/weights/YOLOv3__el1000__opt0.00001__ep0_100.h5')
weight_reader.load_weights(model)
labels = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck","boat"]


def object_detection(model, file_path, class_threshold, labels):
     
    WIDTH, HEIGHT = 416, 416
    
    anchors = [[116, 90, 156,198, 373, 326], [30,61, 62,45, 59, 119], [10,13, 16,30, 33, 23]]

    for file in file_path:
        
        img, image_w, image_h = load_image_pixels(file, (WIDTH, HEIGHT))
        
        yhat = model.predict(img)
        
        boxes = list()
        for i in range(len(yhat)):
            boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, HEIGHT, WIDTH)
   
        correct_yolo_boxes(boxes, image_h, image_w, HEIGHT, WIDTH)

        do_nms(boxes, 0.5)

        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

        for i in range(len(v_boxes)):

            print(v_labels[i], v_scores[i])

        draw_boxes(file, v_boxes, v_labels, v_scores)


img = '756129720103196.jpg'


def get_file_paths(train_folder_with_tamplate):
    
    return glob(train_folder_with_tamplate, recursive=True) 

data = plt.imread(img)
plt.figure(1, figsize=(10,10))    

plt.axis('off')
plt.imshow(data)
plt.show()

object_detection(model, data, 0.7, labels)