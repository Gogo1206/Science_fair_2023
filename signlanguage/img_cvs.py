import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import csv

with open('train.csv', 'w', newline='') as train:

    writer = csv.writer(train, dialect='excel')

    for letter in os.listdir("./signlanguage/tmp/train"):

        print("Iterating: " + letter + " folder")

        for file in tqdm(os.listdir("./emotion/tmp/train/" + letter)):

train.close()