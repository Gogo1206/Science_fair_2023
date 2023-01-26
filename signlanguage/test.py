import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import img_to_array
import os


labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

model = load_model('./signlanguage/models/model3.h5') #3>1>2
# model.summary()

for file in os.listdir("./signlanguage/tmp/test1/"):
    if file.endswith('.jpg'):
        img = Image.open("./signlanguage/tmp/test1/"+file)
        img = img.convert('L')
        # print(img.size)
        img = img.resize((28,28))
        img = img_to_array(img)
        img = img / 255.0
        # plt.imshow(img,cmap='gray')
        # plt.show()
        # plt.waitforbuttonpress
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        # print(prediction)
        text_idx=np.argmax(prediction)
        print("Predicted character: ", labels[text_idx], " with confidence of ", int(prediction[0][text_idx]*100),"%")