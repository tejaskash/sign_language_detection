from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
from imutils import paths
from cv2 import cv2

imagePaths = sorted(list(paths.list_images("dataset")))
labels = list()
for imagePath in imagePaths:
    if imagePath.split("/")[-2] not in labels:
        labels.append(imagePath.split("/")[-2])
print("Labels")
image = cv2.imread("dataset/2/255.jpg")
image = cv2.resize(image,(28,28))
image = image.astype("float") /255.0
image = img_to_array(image)
image = np.expand_dims(image,axis=0)

model = load_model("Model.h5")

preds = model.predict(image)[0]

print(labels[np.argmax(preds)])