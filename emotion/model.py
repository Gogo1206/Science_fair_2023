import numpy as np
import seaborn as sns
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import os
from imblearn import under_sampling, over_sampling
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

df_train = pd.read_csv("./emotion/tmp/metadata_processed.csv")[:28710]
df_test = pd.read_csv("./emotion/tmp/metadata_processed.csv")[28710:]

train_labels = df_train['label']
test_labels = df_test['label']
label_binarizer = LabelBinarizer()
train_labels = label_binarizer.fit_transform(train_labels)
test_labels = label_binarizer.fit_transform(test_labels)

train_img = (df_train.drop(['label'] , axis=1).values / 255.0).reshape(df_train.shape[0], 48, 48, 1)
test_img = (df_test.drop(['label'] , axis=1).values / 255.0).reshape(df_test.shape[0], 48, 48, 1)

target_count = df_train.label.value_counts(sort=False)
# for i in range(7):
#     print("Class " + labels[i] + ":\t", target_count[i])

# test_count = df_test.label.value_counts(sort=False)
# for i in range(7):
#     print("Class " + labels[i] + ":\t", test_count[i])

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

batch_size = 64

train_datagen = ImageDataGenerator(
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    rotation_range = 10,
    # horizontal_flip = True,
    )

train_datagen.fit(train_img)

train_generator = train_datagen.flow(
    train_img,
    train_labels,
    batch_size=batch_size)

model = Sequential([
    Conv2D(512,(3,3), padding='same', input_shape=(48, 48, 1), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    # Dropout(0.25),
    Conv2D(256,(5,5), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    # Dropout(0.25),
    Conv2D(128,(3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    # Dropout(0.25),
    Conv2D(64,(3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    # Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dense(7, activation='softmax'),
])

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

train = model.fit(train_generator, epochs=30, validation_data = (test_img, test_labels), callbacks=[learning_rate_reduction])

model.save("./emotion/models/model.h5")

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

prediction = model.predict(test_img).argmax(axis=-1)
print(prediction[:5])
print(test_labels.argmax(axis=-1)[:5])
print(classification_report(test_labels.argmax(axis=-1), prediction, target_names = ["Class " + labels[i] for i in range(7)]))

cm = confusion_matrix(test_labels.argmax(axis=-1),prediction)
plt.figure(figsize = (15, 15))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='')
plt.show()