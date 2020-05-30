import numpy as np
import os
import pickle


def load_model_and_binarizer(path_outputs, name):
    """
    Params
    ------
    path_outputs : string
        Path to the outputs folder
    name : string
        Name of the model and label binarizer
    """
    # Load model
    path_model = os.path.join(path_outputs, name + ".model")
    model = load_model(path_model)
    
    # Load label binarizer
    path_lb = os.path.join(path_outputs, name + ".pickle")
    lb = pickle.loads(open(path_lb, "rb").read())
    
    return model, lb


def label_image(model, lb, image):
    """
    Params
    ------
    model : Keras model
    lb : Label binarizer
    image : numpy array
        Image to label, properly preprocessed
    """
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    return label