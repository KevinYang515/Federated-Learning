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