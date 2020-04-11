"""
Functions used to train Keras models.
"""

import os
import pickle

from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator


def preprocess_data(data, labels,
                    test_size=0.2, random_state=42):
    """
    Parameters
    ----------
    data : numpy array
        Array containing all the images, in numpy format
        Size (N, height, width, depth), being N the number of images
    labels : numpy array of strings
        Array containg the label assigned to each image
        Size N, being N the number of images
    test_size : float, default 0.2
        Proportion of data that is set apart and only used
        for testing the model
    random_state : integer, default 42
        Random seed, to ease the reproducibility
    """
    
    # Binarize the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    
    # Split data into train and test
    (train_x, test_x,
     train_y, test_y) = train_test_split(data, labels,
                                       test_size=test_size,
                                       random_state=random_state)
    
    return train_x, test_x, train_y, test_y


def train_with_data_augmentation(model, train_x, train_y,
                                 epochs=100, batch_size=32):
    """
    Params
    ------
    model : Keras model
    train_x : Numpy array
    train_y : Numpy array
    epochs : integer, default 100
        Number of epochs to train the model
    batch_size : integer, default 32
        Number of images that are feeded at once
    """
    # Construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    
    # Train the network
    H = model.fit_generator(aug.flow(train_x, train_y, batch_size=batch_size),
                            validation_data=(test_x, test_y),
                            steps_per_epoch=len(train_x) // batch_size,
                            epochs=epochs, verbose=1)
    return model


def store_model_and_binarizer(model, lb, path_outputs, name):
    """
    Params
    ------
    model : Keras model
    lb : Label binarizer
    path_outputs : string
        Path to the outputs folder
    name : string
        Name of the model and label binarizer
    """
    # Store the model
    path_model = os.path.join(path_outputs, name + ".model")
    model.save(path_outputs)
    
    # Store the binarizer
    path_lb = os.path.join(path_outputs, name + ".pickle")
    f = open(path_lb, "wb")
    f.write(pickle.dumps(lb))
    f.close()
