import matplotlib
matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from cnn import LeNet 
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from cv2 import cv2
import os

EPOCHS = 200
INIT_LR = 1e-3
BS = 32
NUM_CLASSES = 2
data = []
labels = []

imagePaths = sorted(list(paths.list_images("dataset")))
random.seed(42)
random.shuffle(imagePaths)
count = 0
labels = dict()
labs = list()
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(28,28))
    image = img_to_array(image)
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    if label not in labels.keys():
        labels[label] = count
        count+=1
    labs.append(labels[label])
data = np.array(data,dtype="float") / 255.0
labels = np.array(labs)

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.25,random_state=42)

trainY = to_categorical(trainY,num_classes=NUM_CLASSES)
testY = to_categorical(testY,num_classes=NUM_CLASSES)

aug = ImageDataGenerator(rotation_range=30,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode="nearest")
model = LeNet.build(width=28,height=28,depth=3,classes=NUM_CLASSES)

opt = Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

model.summary()

H = model.fit(x=aug.flow(trainX,trainY,batch_size=BS),
            validation_data=(testX,testY),steps_per_epoch=len(trainX)//BS,
            epochs=EPOCHS,verbose=1)

model.save("Model.h5")

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("Plot.png")