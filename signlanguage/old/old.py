import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
df_train = pd.read_csv('./signlanguage/tmp/sign_mnist_train.csv')
df_test = pd.read_csv('./signlanguage/tmp/sign_mnist_test.csv')

print(df_train.shape)
print(df_test.shape)

# n_samples = len(df_train.index)
# images = np.array(df_train.drop(['label'],axis=1))
# images = images.reshape(n_samples,28,28)
# plt.figure(figsize=(10,20))
# for i in range(0,50) :
#     plt.subplot(5,10,i+1)
#     plt.axis('off')
#     plt.imshow(images[i], cmap="gray_r")
#     plt.title(labels[df_train.label[i]], loc='center')
# plt.waitforbuttonpress()

train_labels = df_train['label'].values
train_img = df_train.drop(['label'] , axis=1).values.reshape(df_train.shape[0], 28, 28, 1)
test_labels = df_test['label'].values
test_img = df_test.drop(['label'] , axis=1).values.reshape(df_test.shape[0], 28, 28, 1)

#Normalize value
train_img = train_img / 255
test_img = test_img /255

num_classes = 25

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (5,5), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(128, (5,5), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(25, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
train = model.fit(train_img , train_labels , validation_data=(test_img,test_labels), epochs=50, batch_size=125, verbose=1)

model.evaluate(test_img,test_labels)
# print(train.history['accuracy'])
# print(train.history['val_accuracy'])

def plot_scores(train) :
    accuracy = train.history['accuracy']
    val_accuracy = train.history['val_accuracy']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
    plt.plot(epochs, val_accuracy, 'r', label='Score validation')
    plt.title('Scores')
    plt.legend()
    plt.show()

plot_scores(train)

model.save("./signlanguage/model.h5")