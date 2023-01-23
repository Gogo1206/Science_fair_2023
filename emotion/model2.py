import numpy as np
import seaborn as sns
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import os
from imblearn import under_sampling, over_sampling
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from keras.applications import EfficientNetV2L
from PIL import Image

labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

df_train = pd.read_csv("./emotion/tmp/metadata_processed.csv")[:28710]
df_test = pd.read_csv("./emotion/tmp/metadata_processed.csv")[28710:]

train_labels = df_train['label']
test_labels = df_test['label']
label_binarizer = LabelBinarizer()
train_labels = label_binarizer.fit_transform(train_labels)
test_labels = label_binarizer.fit_transform(test_labels)

train_img = df_train.drop(['label'] , axis=1).values.reshape(-1, 48, 48, 1)
test_img = df_test.drop(['label'] , axis=1).values.reshape(-1, 48, 48, 1)

target_count = df_train.label.value_counts(sort=False)
# for i in range(7):
#     print("Class " + labels[i] + ":\t", target_count[i])

def oversampling(train_img = train_img, train_labels = train_labels):
    print(len(train_img))
    train_img = train_img.reshape(train_img.shape[0], -1)
    oversample = over_sampling.RandomOverSampler()
    train_img, train_labels  = oversample.fit_resample(train_img , train_labels)
    train_img = train_img.reshape(-1, 48, 48, 1)
    print(len(train_img))
    return train_img, train_labels

def undersampling():
    angry_count, disgust_count, fear_count, happy_count, neutral_count, sad_count, surprise_count = df_train.label.value_counts(sort=False)

    angry  = df_train[df_train['label'] == 0][:disgust_count]
    disgust = df_train[df_train['label'] == 1][:disgust_count]
    fear = df_train[df_train['label'] == 2][:disgust_count]
    happy = df_train[df_train['label'] == 3][:disgust_count]
    neutral = df_train[df_train['label'] == 4][:disgust_count]
    sad = df_train[df_train['label'] == 5][:disgust_count]
    surprise = df_train[df_train['label'] == 6][:disgust_count]

    return pd.concat([angry, disgust, fear, happy, neutral, sad, surprise], axis=0)

# train_img, train_labels = oversampling()

batch_size = 32

train_datagen = ImageDataGenerator(
    rescale = 1.0/255.0,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    rotation_range = 20,
    horizontal_flip = True,
    )

test_datagen = ImageDataGenerator(
    rescale = 1.0/255.0,
    )

train_datagen.fit(train_img)

train_generator = train_datagen.flow(
    train_img,
    train_labels,
    batch_size=batch_size)

test_generator = test_datagen.flow(
    test_img,
    test_labels,
    batch_size=batch_size)

print(len(train_generator))

model = Sequential([
    EfficientNetV2L(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    Conv2D(1024, 3, 1, activation='relu'),
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dropout(0.2),
    Dense(1024, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax'),
])

for layer in model.layers[:-6]:
    layer.trainable = False

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

train = model.fit(train_generator, epochs=50, validation_data = test_generator, callbacks=[learning_rate_reduction], steps_per_epoch=28710/32)

model.save("./emotion/models/model2.h5")

def plot_results(train):
    acc = train.history['accuracy']
    val_acc = train.history['val_accuracy']
    loss = train.history['loss']
    val_loss = train.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.figure(figsize = (24, 6))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()
 
# print best epoch with best accuracy on validation

def get_best_epcoh(train):
    valid_acc = train.history['val_accuracy']
    best_epoch = valid_acc.index(max(valid_acc)) + 1
    best_acc =  max(valid_acc)
    print('Best Validation Accuracy Score {:0.5f}, is for epoch {}'.format( best_acc, best_epoch))
    return best_epoch

best_epoch = get_best_epcoh(train)
plot_results(train)