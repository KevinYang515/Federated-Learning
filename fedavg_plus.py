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
import cv2
import random
import data.preprocessing as prep
from data.read_data import read_data, read_setting
from data.data_utils import load_cifar10_data, train_test_label_to_categorical
from model.model import init_model, record_history
from model.operation import broadcast_to_device, caculate_delta, aggregate_add, aggregate_division_return 

from math import floor
from sklearn.utils import shuffle
from random import randint
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
    # Read detailed settings from json file
    detailed_setting = read_setting()
    # Read data from input file
    data_distribution = read_data(detailed_setting["file_source"]["data_distribution_file"], detailed_setting["file_source"]["data_information_file"])
    # Load CIFAR-10 data
    train_images, train_labels, test_images, test_labels = load_cifar10_data()
    # Transfer train and test label to be categorical
    train_label, test_label = train_test_label_to_categorical(train_labels, test_labels)

    # adjust parameters of the model
    callback = tf.keras.callbacks.LearningRateScheduler(step_decay)
    # define the method for preprocessing
    augment = ImageDataGenerator(preprocessing_function=prep.preprocessing_for_training)
    
    # device list for demand 10000
    device_list = detailed_setting["device_list"]["10000"]

    new_device_info = {}
    for x in device_list:
        new_device_info[x] = data_distribution[x]

    training_info = detailed_setting["training_info"]

    # Define our model
    model_m, history_total = init_model(True)

    for _ in range(training_info["num_round"]):
        print("\n" + "\033[1m" + "Round: " + str(_))
        for device in new_device_info:
            print("\033[0m" + "Device:", device, "model_" + str(device))

            if (_ == 0):
                #Define and initialize an estimator model
                locals()['model_{}'.format(device)] = init_model(False)
            else:
                #Broadcast to every device (e.g., model_m parameters to all devices)
                broadcast_to_device(locals()['model_{}'.format(device)], model_m)

            #Local training on each device 
            for epo in range(training_info["num_local_epoch"]):
                train_image_temp, train_label_temp = prep.prepare_for_training_data0(train_images, train_labels, data_distribution, device)
                train_image_crop = np.stack([prep.random_crop(train_image_temp[i], 24, 24) for i in range(len(train_image_temp))], axis=0)
                train_new_image, train_new_label = shuffle(train_image_crop, 
                                                        train_label_temp, 
                                                        random_state=randint(0, train_image_crop.shape[0]))

                history_temp = locals()['model_{}'.format(device)].fit_generator(
                    augment.flow(train_new_image, train_new_label, batch_size=training_info["local_batch_size"]),
                    epochs=1, 
                    callbacks=[callback],
                    verbose=training_info["show"])

            #Calculate delta weight on each device
            caculate_delta(locals()['model_{}'.format(device)], model_m)

        #Aggregate all delta weight on device 0 (Addition)
        for device in device_list[1:]:
            aggregate_add(locals()['model_{}'.format(device)], locals()['model_{}'.format(device_list[0])])

        #Aggregate all delta weight on device 0 (Division) and return total delta weight to center device
        aggregate_division_return(locals()['model_{}'.format(device_list[0])], model_m, len(device_list))

        print("Result : " + str(_))
        #Evaluate with new weight
        test_d = np.stack([prep.preprocessing_for_testing(test_images[i]) for i in range(10000)], axis=0)
        test_new_image, test_new_label = shuffle(test_d, test_label, 
                                                random_state=randint(1, train_images.shape[0]))

        history_temp = model_m.evaluate(test_new_image, test_new_label, batch_size=64)

        #Record each round accuracy
        record_history(history_temp, history_total)

    print("===============Finish===============")
    print("Accuracy: ")
    print(history_total["val_acc"])
    print("Loss: ")
    print(history_total["val_loss"])

if __name__ == '__main__':
    main(sys.argv)