"""
This model is based on the one designed by Adrian Rosebrock
Link:
https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/
"""

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


def build_model(width, height, depth, classes):
    """
    Builds a keras CNN model ready to be trained.

    Parameters
    ----------
    width : integer
        Width of the input images
    height : integer
        Height of the input images
    depth : integer
        Number of color channels
    classes : integer
        Number of categories to predict
    """
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    # if we are using "channels first", update the input shape
    # and channels dimension
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    # CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3))) # máximo de los 9
    model.add(Dropout(0.25))

    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # first (and only) set of FullyConnected => RELU layers
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model


def initialize_model(lb,
                     image_dims=(96, 96, 3),
                     init_lr=1e-3,
                     epochs=100,
                     loss="categorical_crossentropy",
                     metrics=["accuracy"]):
    """
    Initializes Rosebrock model.
    
    Params
    ------
    lb : Label binarizer
    init_lr : float
        Starting learning rate
    epochs : integer
        Number of epochs to train the model
    loss : string
        Loss function
    metrics : array of strings
        Score metrics to show
        
    Returns
    -------
    model : Keras model
        CNN ready to be feeded and trained
    """
    model = build_model(width=image_dims[1], 
                        height=image_dims[0], 
                        depth=image_dims[2], 
                        classes=len(lb.classes_))
    opt = Adam(lr=init_lr, decay=init_lr / epochs)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    return model
