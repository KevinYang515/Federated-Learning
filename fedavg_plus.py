# Modified fedavg for our input

# Set Specific GPU in tensorflow
# We can check GPU information with command ```nvidia-smi```.
# If there is only one GPU avaliable, we set the os environ to "0".
# If there are two GPU avaliable and we would like to set the second GPU, 
#   we set the os environ to "1".
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Other import
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import data/preprocessing as prep
import model/model

from math import floor
from sklearn.utils import shuffle
from random import randint
from tensorflow.keras import datasets, layers, models
from keras.utils import to_categorical
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

# We can make sure which specific GPU, and avoid crashing because of out of GPU memory.
with K.tf.device('/device:GPU:0'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def main(argv):
    with open('./mypaper/data_distribution_3000_new_14.txt', 'r') as rf:
        device_information = rf.readlines()
        
    with open('./mypaper/device_info_3000.txt', 'r') as rf:
        device_information_detail = rf.readlines()

    data_distribution = {}
    each_class = []

    for i in range(10):
        each_class.append(0)
        
    for i in range(len(device_information)):
        temp = []
        temp_class = device_information[i].split(';')[1][1:-1].split(' ')
        temp_emd = device_information[i].split(';')[0]
        temp_info = device_information_detail[i][1:-2].split(',')
        temp_var = device_information[i].split(';')[2][:-1]
        
        for j in range(10):
            temp.append([each_class[j], each_class[j] + int(temp_class[j])])
            each_class[j] += int(temp_class[j])
            
        data_distribution[i] = {}
        data_distribution[i]['training time'] = int(temp_info[0])
        data_distribution[i]['transmission time'] = float(temp_info[1])
        data_distribution[i]['data_quantity'] = int(temp_info[2])
        data_distribution[i]['emd'] = float(temp_emd)
        data_distribution[i]['variance'] = float(temp_var)
        data_distribution[i]['data_distribution'] = temp


    # If we haven't downloaded cifar10 dataset, it will automatically download it for us.
    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    train_label = to_categorical(train_labels)
    test_label = to_categorical(test_labels)

if __name__ == '__main__':
    app.run(main)