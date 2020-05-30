import os

from imutils import paths
from sklearn.preprocessing import LabelBinarizer

from cloud_happens.preprocessing.preprocessing import load_data_and_labels
from cloud_happens.model.train import preprocess_data
from cloud_happens.model.rosebrock import initialize_model
from cloud_happens.model.train import train_with_data_augmentation
from cloud_happens.model.train import store_model_and_binarizer

path_data = "../data"
path_outputs = "outputs"

image_dims = (96, 96)

test_size = 0.2
epochs = 100
batch_size = 64
lr = 0.001

if not os.path.exists(path_outputs):
    os.mkdir(path_outputs)

"""
Load and preprocess
"""

# List the images contained in the data folder
image_paths = list(paths.list_images(path_data))

data, labels = load_data_and_labels(image_paths, image_dims=image_dims)

lb = LabelBinarizer()
labels_bin = lb.fit_transform(labels)

"""
Train
"""

image_dims = image_dims + (3, )

model = initialize_model(lb, image_dims=image_dims, init_lr=lr)

train_x, test_x, train_y, test_y = preprocess_data(data, labels_bin,
                                                   test_size=test_size)

model = train_with_data_augmentation(model, train_x, train_y, test_x, test_y,
                                     epochs=epochs, batch_size=batch_size)

store_model_and_binarizer(model, lb, path_outputs, "rosebrock")
