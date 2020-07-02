import cv2
import random

from imutils import resize
from imutils import paths

from cloud_happens.preprocessing.preprocessing import load_data_and_labels
from cloud_happens.model.test import load_model_and_binarizer

path_data = "data"
path_outputs = "outputs"

sample = 10

"""
Load model and data
"""

# Load model and label binarizer
model, lb = load_model_and_binarizer(path_outputs, 'rosebrock')

# Get image_dims according to the first layer of the model
image_dims = model.layers[0].input_shape[1:3]

# List the images contained in the data folder
image_paths = list(paths.list_images(path_data))
sample = min(len(image_paths), sample)

data, labels = load_data_and_labels(image_paths, image_dims=image_dims)

"""
Test model
"""
pred = model.predict(data)
pred_labels = lb.inverse_transform(pred)

for idx in random.sample(range(data.shape[0]), sample):
    img = data[idx, :, :, :]
    lab = labels[idx]
    pred_lab = pred_labels[idx]

    # pred = model.predict(img[np.newaxis, ...])
    # pred = lb.inverse_transform(pred)[0]

    # Correct BGR to RGB
    img = img[:, :, ::-1]
    # Scale image to ease its view
    img = resize(img, width=720)
    # Tag real and predicted labels
    tag = f"Real: {lab}"
    cv2.putText(img, tag, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    tag = f"Pred: {pred_lab}"
    cv2.putText(img, tag, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    cv2.imshow('CloudHappens', img)
    key = cv2.waitKey(2000)
    if key == 27:  # if ESC is pressed, exit loop
        cv2.destroyAllWindows()
        break
