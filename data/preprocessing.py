import cv2
import numpy as np
from sklearn.utils import shuffle
from random import randint
from keras.utils import to_categorical
import random

def random_contrast(im, lower=0.2, upper=1.8):
    prob = randint(0, 1)
    if prob == 1:
        alpha = random.uniform(lower, upper)
        imgg = im * alpha
        imgg = imgg.clip(min=0, max=1)
        return imgg
    else:
        return im

def random_bright(im, delta=63):
    prob = randint(0,1)
    if prob == 1:
        delta = random.uniform(-delta, delta)
        imgg = im + delta / 255.0
        imgg = imgg.clip(min=0, max=1)
        return imgg
    else:
        return im

def per_image_standardization(img):
    num_compare = img.shape[0] * img.shape[1] * 3
    img_arr=np.array(img)
    img_t = (img_arr - np.mean(img_arr))/max(np.std(img_arr), 1/num_compare)
    return img_t

def random_crop(img, width, height):
    width1 = randint(0, img.shape[0] - width)
    height1 = randint(0, img.shape[1] - height)
    cropped = img[height1:height1+height, width1:width1+width]

    return cropped

def random_flip_left_right(image):
    prob = randint(0, 1)
    if prob == 1:
        image = np.fliplr(image)
    return image

def preprocessing_for_training(images):
    distorted_image = random_flip_left_right(images)
    distorted_image = random_bright(distorted_image)
    distorted_image = random_contrast(distorted_image)
    float_image = per_image_standardization(distorted_image)
    
    return float_image

def preprocessing_for_testing(images):
    distorted_image = cv2.resize(images, dsize=(24, 24), interpolation=cv2.INTER_CUBIC)
    distorted_image = per_image_standardization(distorted_image)
    
    return distorted_image

# Seperate data for each device
def prepare_for_training_data0(device_num):
    # Return
    image, label = train_images, train_labels
    all_class_device = data_distribution[device_num]

    device_num_start = data_distribution[device_num]['data_distribution'][0][0] % 5000
    device_num_end = data_distribution[device_num]['data_distribution'][0][1] % 5000

    if device_num_start > device_num_end: 
        a = image[label[:, 0] == 0][device_num_start:]
        b = image[label[:, 0] == 0][:device_num_end]

        a_label = label[label[:, 0] == 0][device_num_start:]
        b_label = label[label[:, 0] == 0][:device_num_end]
        s0 = [np.vstack((a,b)), np.vstack((a_label,b_label))]
    else:
        s0 = [image[label[:, 0] == 0][device_num_start : device_num_end], label[label[:, 0] == 0][device_num_start : device_num_end]]

    for i in range(1, 10):
        device_num_start = data_distribution[device_num]['data_distribution'][0][0] % 5000
        device_num_end = data_distribution[device_num]['data_distribution'][0][1] % 5000

        if device_num_start > device_num_end: 
            a = image[label[:, 0] == i][device_num_start:]
            b = image[label[:, 0] == i][:device_num_end]

            a_label = label[label[:, 0] == i][device_num_start:]
            b_label = label[label[:, 0] == i][:device_num_end]
            s1 = [np.vstack((a,b)), np.vstack((a_label,b_label))]
        else:
            s1 = [image[label[:, 0] == i][device_num_start : device_num_end], label[label[:, 0] == i][device_num_start : device_num_end]]

        s0 = [np.concatenate((s0[0], s1[0]), axis=0), np.append(s0[1], s1[1])]


    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
    
    return s0[0], to_categorical(s0[1])


def prepare_for_training_data(device_num, train_images, train_labels, num_device):
    num_data = int(len(train_images)/num_device/10)
    device_num = device_num * num_data
    
    image, label = train_images, train_labels
    
    s0 = [image[label[:, 0] == 0][device_num : device_num+num_data], label[label[:, 0] == 0][device_num : device_num+num_data]]

    for i in range(1, 10):
        s1 = [image[label[:, 0] == i][device_num : device_num+num_data], label[label[:, 0] == i][device_num : device_num+num_data]]

        s0 = [np.concatenate((s0[0], s1[0]), axis=0), np.append(s0[1], s1[1])]

    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
    
    return s0[0], to_categorical(s0[1])

def prepare_for_testing_data(device_num, test_images, test_labels, num_device):
    num_data = int(len(test_images)/num_device/10)
    device_num = device_num * num_data

    image, label = test_images, test_labels
#     image, label = shuffle(test_images, test_labels, random_state=0)
    
    s0 = [image[label[:, 0] == 0][device_num : device_num+num_data], label[label[:, 0] == 0][device_num : device_num+num_data]]

    for i in range(1, 10):
        s1 = [image[label[:, 0] == i][device_num : device_num+num_data], label[label[:, 0] == i][device_num : device_num+num_data]]

        s0 = [np.concatenate((s0[0], s1[0]), axis=0), np.append(s0[1], s1[1])]

    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
    
    return s0[0], to_categorical(s0[1])

def prepare_for_training_data_mnist(device_num, train_images, train_labels, num_device):
    num_data = int(len(train_images)/num_device/10)
    device_num = device_num * num_data

    image, label = train_images, train_labels
    
    s0 = [image[label == 0][device_num : device_num + num_data], label[label == 0][device_num : device_num + num_data]]

    for i in range(1, 10):
        s1 = [image[label == i][device_num : device_num+num_data], label[label == i][device_num : device_num+num_data]]

        s0 = [np.concatenate((s0[0], s1[0]), axis=0), np.append(s0[1], s1[1])]

    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
    
    return s0[0], to_categorical(s0[1], 10)


def prepare_for_testing_data_mnist(device_num, test_images, test_labels, num_device):
    num_data = int(len(test_images)/num_device/10)
    device_num = device_num * num_data

    image, label = test_images, test_labels
#     image, label = shuffle(test_images, test_labels, random_state=0)
    
    s0 = [image[label == 0][device_num : device_num+num_data], label[label == 0][device_num : device_num+num_data]]

    for i in range(1, 10):
        s1 = [image[label == i][device_num : device_num+num_data], label[label == i][device_num : device_num+num_data]]

        s0 = [np.concatenate((s0[0], s1[0]), axis=0), np.append(s0[1], s1[1])]

    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
    
    return s0[0], to_categorical(s0[1], 10)
