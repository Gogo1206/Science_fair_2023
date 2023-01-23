import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras import utils
import cv2
import glob
import os

model = tf.keras.models.load_model('./practice/rps.h5')
# model.summary()

uploaded = [cv2.imread(file) for file in glob.glob("./practice/tmp/img/*.png")]

for fn in uploaded:
    path = fn
    # predicting images
    img = cv2.resize(path,(150,150))
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    x = utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    # print(fn)
    print(classes)