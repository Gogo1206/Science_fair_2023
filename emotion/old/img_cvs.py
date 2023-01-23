import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np


columnNames = list()
columnNames.append('label')
for i in range(784):
    pixel = str(i)
    columnNames.append(pixel)

train_data = pd.DataFrame(columns = columnNames)

num_images = 0
i = 0

for expression in os.listdir("./emotion/tmp/train"):

    print("Iterating: " + expression + " folder")

    for file in tqdm(os.listdir("./emotion/tmp/train/" + expression)):
        img = Image.open("./emotion/tmp/train/" + expression + "/" + file)
        img = img.resize((28, 28))
        img.load()
        imgdata = np.asarray(img, dtype="int32")
        data = []
        data.append(i)
        for x in range(28):
            for y in range(28):
                data.append(imgdata[x][y])
        train_data.loc[num_images] = data
        num_images += 1
    
    i = i + 1

train_data.to_csv("./emotion/tmp/train.csv", index=False)


test_data = pd.DataFrame(columns = columnNames)

num_images = 0
i = 0

for expression in os.listdir("./emotion/tmp/validation"):

    print("Iterating: " + expression + " folder")

    for file in tqdm(os.listdir("./emotion/tmp/validation/" + expression)):
        img = Image.open("./emotion/tmp/validation/" + expression + "/" + file)
        img = img.resize((28, 28))
        img.load()
        imgdata = np.asarray(img, dtype="int32")
        data = []
        data.append(i)
        for x in range(28):
            for y in range(28):
                data.append(imgdata[x][y])
        test_data.loc[num_images] = data
        num_images += 1
    
    i = i + 1

test_data.to_csv("./emotion/tmp/test.csv", index=False)