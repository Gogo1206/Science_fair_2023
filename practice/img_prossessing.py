import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

def convert(value, type): #procedure
    r, g, b = value
    if(type=="gray"):
        return 0.299*r+0.587*g+0.114*b # convert to gray
    if(type=="red"):
        return [r, 0, 0] # convert to red
    if(type=="green"):
        return [0, g, 0] # convert to green
    if(type=="blue"):
        return [0, 0, b] # convert to blue
    return value

img = Image.open(r"./practice/tmp/image.jpg") #input
plt.imshow(img)
plt.show()
img_transformed = np.copy(img) #list (ndarray)
for x in tqdm(range(img_transformed.shape[0])): #iteration
    for y in range(img_transformed.shape[1]):
        img_transformed[x, y] = convert(img_transformed[x, y], "gray") #call to procedure
plt.imshow(img_transformed) #output
plt.show()
img_transformed = np.copy(img) #list (ndarray)
for x in tqdm(range(img_transformed.shape[0])): #iteration
    for y in range(img_transformed.shape[1]):
        img_transformed[x, y] = convert(img_transformed[x, y], "red") #call to procedure
plt.imshow(img_transformed) #output
plt.show()
img_transformed = np.copy(img) #list (ndarray)
for x in tqdm(range(img_transformed.shape[0])): #iteration
    for y in range(img_transformed.shape[1]):
        img_transformed[x, y] = convert(img_transformed[x, y], "green") #call to procedure
plt.imshow(img_transformed) #output
plt.show()
img_transformed = np.copy(img) #list (ndarray)
for x in tqdm(range(img_transformed.shape[0])): #iteration
    for y in range(img_transformed.shape[1]):
        img_transformed[x, y] = convert(img_transformed[x, y], "blue") #call to procedure
plt.imshow(img_transformed) #output
plt.show()