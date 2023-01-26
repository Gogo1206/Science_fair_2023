import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

df_train = pd.read_csv('./signlanguage/tmp/sign_mnist_train.csv')
df_test = pd.read_csv('./signlanguage/tmp/sign_mnist_test.csv')
train_labels = df_train['label']
test_labels = df_test['label']
label_binarizer = LabelBinarizer()
train_labels = label_binarizer.fit_transform(train_labels)
test_labels = label_binarizer.fit_transform(test_labels)
train_img = (df_train.drop(['label'] , axis=1).values / 255.0).reshape(-1, 28, 28, 1)
test_img = (df_test.drop(['label'] , axis=1).values / 255.0).reshape(-1, 28, 28, 1)

print(df_train.shape)
print(df_test.shape)

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# n_samples = len(df_train.index) # view first 50 image
# images = np.array(df_train.drop(['label'],axis=1))
# images = images.reshape(n_samples,28,28)
# plt.figure(figsize=(10,20))
# for i in range(2,52) :
#     plt.subplot(5,10,i+1)
#     plt.axis('off')
#     plt.imshow(images[i], cmap="gray_r")
#     plt.title(labels[df_train.label[i]], loc='center')
# plt.waitforbuttonpress()

datagen = ImageDataGenerator( # Data Augmentation
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.1, # Randomly zoom image 
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    # horizontal_flip=True,
    # brightness_range=[0.7,1.5]
)
datagen.fit(train_img)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)
model = Sequential([
    Conv2D(256, (3,3), strides = 1, padding = 'same', activation = 'relu', input_shape = (28,28,1)),
    Dropout(0.25),
    BatchNormalization(),
    # MaxPooling2D((2,2), strides = 2, padding = 'same'),
    Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'),
    Dropout(0.25),
    BatchNormalization(),
    Conv2D(64, (3,3), strides = 1, padding = 'same', activation = 'relu'),
    Dropout(0.25),
    BatchNormalization(),
    Flatten(),
    Dense(units = 4096, activation = 'relu'),
    Dropout(0.25),
    Dense(units = 256, activation = 'relu'),
    Dropout(0.25),
    Dense(units = 26, activation = 'softmax'),
])

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

train = model.fit(datagen.flow(train_img, train_labels, batch_size = 64), epochs = 15, validation_data = (test_img, test_labels), callbacks = [learning_rate_reduction])

model.save("./signlanguage/models/model3.h5")

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
 
plot_results(train)
# print best epoch with best accuracy on validation

def get_best_epcoh(train):
    valid_acc = train.history['val_accuracy']
    best_epoch = valid_acc.index(max(valid_acc)) + 1
    best_acc =  max(valid_acc)
    print('Best Validation Accuracy Score {:0.5f}, is for epoch {}'.format( best_acc, best_epoch))
    return best_epoch

best_epoch = get_best_epcoh(train)

prediction = model.predict(test_img).argmax(axis=-1)
print(classification_report(test_labels.argmax(axis=-1), prediction, target_names = ["Class " + labels[i] for i in range(len(labels))]))

cm = confusion_matrix(test_labels.argmax(axis=-1),prediction)
plt.figure(figsize = (15, 15))
sns.heatmap(cm,cmap= "flare", linecolor = 'black' , linewidth = 0 , annot = False, fmt='')
plt.show()