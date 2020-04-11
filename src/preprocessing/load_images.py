"""
These functions are used to load the images
in numpy array format.
"""
import cv2
import random

from imutils import paths


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


def load_data_and_labels(image_paths):
    """
    Take given list of image paths and load each image (as a numpy array)
    and it's corresponding label (as a string)
    
    Returns
    -------
    data : numpy array
        Array containing all the images, in numpy format
        Size (N, height, width, depth), being N the number of images
    labels : numpy array of strings
        Array containg the label assigned to each image
        Size N, being N the number of images
    """
    # Initialize arrays
    data = []
    labels = []

    for image_path in image_paths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the labels list
        label = image_path.split(os.path.sep)[-2]
        labels.append(label)
    
    # Scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    
    return data, labels
