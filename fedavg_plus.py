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

    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0


    train_label = to_categorical(train_labels)
    test_label = to_categorical(test_labels)

if __name__ == '__main__':
    app.run(main)