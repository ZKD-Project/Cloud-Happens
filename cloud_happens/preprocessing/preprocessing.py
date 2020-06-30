"""
These functions are used to load the images
in numpy array format.
"""
import cv2
import numpy as np
import os
import random

from imutils import paths
from keras.preprocessing.image import img_to_array


def list_image_paths(path_data, random_seed=None):
    """
    List all image paths contained in give root folder.
    If provided random_seed, shuffle them.
    """
    image_paths = sorted(list(paths.list_images(path_data)))
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(image_paths)
    return image_paths


def load_data_and_labels(image_paths, image_dims=None, verbose=True):
    """
    Take given list of image paths and load each image (as a numpy array)
    and it's corresponding label (as a string)

    Params
    ------
    image_paths : list
        List of the image paths (strings).
    image_dims : tuple, default=None
        If given, resize images to image_dims (width x height).
    verbose : bool, default=True
        Print process info
    
    Returns
    -------
    data : numpy array
        Array containing all the images, in numpy format
        Size (N, height, width, depth), being N the number of images
    labels : numpy array of strings
        Array containing the label assigned to each image
        Size N, being N the number of images
    """
    # Initialize arrays
    data = []
    labels = []

    if (image_dims is not None) and (len(image_dims) != 2):
        raise ValueError("image_dims parameter must be a tuple of two "
                         f"elements.\nimage_dims = {image_dims}")

    if verbose:
        num_images = len(image_paths)
        print(f"Loading {num_images} images")

    for idx, image_path in enumerate(image_paths):
        if not os.path.isfile(image_path):
            raise ValueError(f"Path is not a file:\n{image_path}")
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image_dims is not None:
            image = cv2.resize(image, (image_dims[1], image_dims[0]))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and
        # update the labels list
        label = image_path.split(os.path.sep)[-2]
        labels.append(label)

        if verbose:
            print(f" Image {idx} out of {num_images}")
    
    # Scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    
    return data, labels


def load_single_image(path_image, image_dims=None):
    """
    Load an image and preprocess it
    
    Params
    ------
    path_image : string
        Path to the image
    image_dims : tuple, default=None
        If given, resize images to image_dims (width x height).
    
    Returns
    -------
    image : numpy array
        Tensor ready to be labeled by the model
    """

    if not os.path.isfile(path_image):
        raise ValueError(f"Path is not a file:\n{path_image}")

    image = cv2.imread(path_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image_dims is not None:
        if len(image_dims) != 2:
            raise ValueError("image_dims parameter must be a tuple of two "
                             f"elements.\nimage_dims = {image_dims}")
        image = cv2.resize(image, (image_dims[1], image_dims[0]))

    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image
