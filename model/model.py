import tensorflow as tf
from tensorflow.keras import layers, models

def init_model(is_master=False):
    """
    Initial model, compile, and load initial weight
    If we want the master one, it will return model and history for storing accuracy and loss.
    if we want the client one, it will return only model.
    :param check the model we want master or client
    :return initialized, compiled, loaded model
    :return history dictionary (optional)
    """
    model_x = define_model()
    init_compile(model_x)
    init_load_weight(model_x)
    if is_master == True:
        return model_x, init_history()
    else:
        return model_x

# Define our model
def define_model():
    """
    Define the architecture of our model from the paper of Google
    :return model architecture 
    """
    return tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), padding='same', activation='relu', input_shape=(24, 24, 3)),
        layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (5,5), padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(384, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(192, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

def init_compile(model_x):
    """
    Compile the input model
    :param model which we want to be compiled
    """
    model_x.compile(optimizer='sgd',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

def init_load_weight(model_x):
    """
    Load initial weight for the input model
    :param model which we want to load the initial weight
    """
    model_x.load_weights('model/weight/ini_cifar10.h5')

def init_history():
    """
    Return the inital history dictionary to store accuracy and loss
    :return history dictionary
    """
    history = {}
    history['val_loss'] = [2.3840]
    history['val_acc'] = [0.0976]
    return history

def record_history(history_temp, history_total):
    """
    Record accuracy and loss result to history_total
    :param the dictionary stored the accuracy and loss for this round
    :param the dictionary stored the accuracy and loss for all round 
    """
    history_total['val_loss'].append(history_temp[0])
    history_total['val_acc'].append(history_temp[1])