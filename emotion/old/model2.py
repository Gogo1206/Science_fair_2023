import tensorflow as tf
tf.__version__
from keras.layers import Lambda,Input,Dense,Flatten
from keras.models import Model
from keras.applications.efficientnet import EfficientNetB2
from keras.applications.efficientnet import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img
from keras.models import Sequential
import numpy as np
from glob import glob

IMAGE_SIZE=[224,224]

train_path = "./emotion/tmp/train/"
valid_path = "./emotion/tmp/validation/"

eff=EfficientNetB2(input_shape=IMAGE_SIZE+[3],include_top=False)

folders=glob('./emotion/tmp/train/*')

for layer in eff.layers:
  layer.trainable=False

x=Flatten()(eff.output)

prediction=Dense(len(folders),activation='softmax')(x)

model=Model(inputs=eff.input,outputs=prediction)

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory(train_path,
                                               target_size=(224,224),batch_size=32,class_mode='categorical')

test_set=test_datagen.flow_from_directory(valid_path,
                                               target_size=(224,224),batch_size=32,class_mode='categorical')

r=model.fit(training_set,validation_data=test_set,
                      epochs=3,
                      steps_per_epoch=len(training_set),
                      validation_steps=len(test_set))