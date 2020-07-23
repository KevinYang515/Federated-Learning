# Modified fedavg for our input
from __future__ import absolute_import, division, print_function, unicode_literals

# Set Specific GPU in tensorflow
# We can check GPU information with command ```nvidia-smi```.
# If there is only one GPU avaliable, we set the os environ to "0".
# If there are two GPU avaliable and we would like to set the second GPU, 
#   we set the os environ to "1".
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Other import
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import data.preprocessing as prep
from data.read_data import read_data
from data.data_utils import load_cifar10_data, train_test_label_to_categorical
from model.model import define_model, init_compile

from math import floor
from sklearn.utils import shuffle
from random import randint
from tensorflow.keras import layers, models
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

# We can make sure which specific GPU, and avoid crashing because of out of GPU memory.
with K.tf.device('/device:GPU:0'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
_, epo = 0, 0

def step_decay(epoch):
    epoch = _ + epo
    # initial_lrate = 1.0 # no longer needed
    drop = 0.99
    epochs_drop = 1.0
    
    lrate = 0.2 * pow(drop, floor((1+epoch)/epochs_drop))
    return lrate

def main(argv):
    data_distribution_file = './data/file/data_distribution_3000_new_14.txt'
    data_information_file = './data/file/device_info_3000.txt'

    # Read data from input file
    data_distribution = read_data(data_distribution_file, data_information_file)
    # Load CIFAR-10 data
    train_images, train_labels, test_images, test_labels = load_cifar10_data()
    # Transfer train and test label to be categorical
    train_label, test_label = train_test_label_to_categorical(train_labels, test_labels)

    # adjust parameters of the model
    callback = tf.keras.callbacks.LearningRateScheduler(step_decay)
    
    augment = ImageDataGenerator(preprocessing_function=prep.preprocessing_for_training)
    history_02 = {}
    # 30000
    # device_list = [0, 20, 21, 22, 23, 27, 37, 65, 73, 81, 95, 98]

    # 10000
    device_list = [0, 20, 22, 23, 81]

    new_device_info = {}

    for x in device_list:
        new_device_info[x] = data_distribution[x]

    num_device = 100
    num_center_epoch = 1
    num_local_epoch = 5
    num_round = 2
    center_batch_size = 64
    local_batch_size = 50

    show = 0

    # Define our model
    model_m = define_model()
    init_compile(model_m)

    #Initialization for all device's record
    history_m = {}
    for device in new_device_info:
        locals()['history_{}'.format(device)] = {}

    test_d = np.stack([prep.preprocessing_for_testing(test_images[i]) for i in range(10000)], axis=0)
    history_temp = model_m.evaluate(test_d, test_label, batch_size=64)

    # try and error history_m
    history_02['val_loss'] = [history_temp[0]]
    history_02['val_acc'] = [history_temp[1]]

    for _ in range(num_round):
        print("\n" + "\033[1m" + "Round: " + str(_))
        for device in new_device_info:
            print("\033[0m" + "\nDevice:", device, "model_" + str(device))

            if(_ == 0):
                #Define and initialize an estimator model
                locals()['model_{}'.format(device)] = define_model()
                init_compile(locals()['model_{}'.format(device)])
                
            #Broadcast to every device (e.g., model_m parameters to all devices)
            for layer in range(len(model_m.layers)):
                locals()['model_{}'.format(device)].layers[layer].set_weights(model_m.layers[layer].get_weights())

            #Local training on each device 
            for epo in range(num_local_epoch):
                print("Epoch:", epo)
                train_image_temp, train_label_temp = prep.prepare_for_training_data0(train_images, train_labels, data_distribution, device)
                train_image_crop = np.stack([prep.random_crop(train_image_temp[i], 24, 24) for i in range(len(train_image_temp))], axis=0)

                test_image_temp, test_label_temp = prep.prepare_for_testing_data(test_images, test_labels, device, len(new_device_info))
                test_image_crop = np.stack([prep.preprocessing_for_testing(test_image_temp[i]) for i in range(len(test_image_temp))], axis=0)

                train_new_image, train_new_label = shuffle(train_image_crop, 
                                                        train_label_temp, 
                                                        random_state=randint(0, train_image_crop.shape[0]))
                test_new_image, test_new_label = shuffle(test_image_crop, 
                                                        test_label_temp, 
                                                        random_state=randint(0, test_image_crop.shape[0]))

    #             show = 1 if e == num_device else 0

                history_temp = locals()['model_{}'.format(device)].fit_generator(
                    augment.flow(train_new_image, train_new_label, batch_size=local_batch_size), 
                    epochs=1, 
                    callbacks=[callback],
                    verbose=show,
                    validation_data=(test_new_image, test_new_label))

                #Record accuracy of each device
                for attribute in history_temp.history:
                    if attribute in locals()['history_{}'.format(device)]:
                        locals()['history_{}'.format(device)][attribute].extend(history_temp.history[attribute])
                    else:
                        locals()['history_{}'.format(device)][attribute] = history_temp.history[attribute]


            #Calculate delta weight on each device
            for layer in range(len(model_m.layers)):
                locals()['model_{}'.format(device)].layers[layer].set_weights(
                    np.subtract(locals()['model_{}'.format(device)].layers[layer].get_weights(),
                    model_m.layers[layer].get_weights()))
                
        #Aggregate all delta weight on device 0 (Addition)
        for device in [x for x in new_device_info][1:]:
            for layer in range(len(model_m.layers)):
                model_0.layers[layer].set_weights(
                    np.add(locals()['model_{}'.format(device)].layers[layer].get_weights(),
                    model_0.layers[layer].get_weights()))

        #Aggregate all delta weight on device 0 (Division)
        for layer in range(len(model_m.layers)):
            model_0.layers[layer].set_weights(np.divide(model_0.layers[layer].get_weights(), len(new_device_info)))

        #Return total delta weight to center device
        for layer in range(len(model_m.layers)):
            model_m.layers[layer].set_weights(np.add(model_m.layers[layer].get_weights(),
                                                    model_0.layers[layer].get_weights()))
        print("Result : " + str(_))
        #Evaluate with new weight
        test_d = np.stack([prep.preprocessing_for_testing(test_images[i]) for i in range(10000)], axis=0)
        test_new_image, test_new_label = shuffle(test_d, test_label, 
                                                random_state=randint(1, train_images.shape[0]))
        
        history_temp = model_m.evaluate(test_new_image, test_new_label, batch_size=64)

        #Record each round accuracy
        history_02['val_loss'].append(history_temp[0])
        history_02['val_acc'].append(history_temp[1])

if __name__ == '__main__':
    main(sys.argv)