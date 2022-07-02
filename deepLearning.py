# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:26:01 2021

@author: YALÇINKAYA
"""

import tensorflow as tf

from tensorflow.keras.models import Sequential          # ardışıl CNN modelleri oluşturmak için
from tensorflow.keras.layers import Flatten, Dense      # CNN katmanları için
from tensorflow.keras.optimizers import RMSprop,Adam    # öğrenme algoratması
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import keras

img_dim = (224,224,3)
ind = 4
vgg16_model, inception_model, resnet_model, xception_model, mobilenet_model = Sequential(),Sequential(),Sequential(),Sequential(),Sequential()
if ind==0:
    vgg16_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=img_dim)
elif ind==1:
    inception_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=img_dim)
elif ind==2:    
    resnet_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=img_dim)
elif ind==3:    
    xception_model = tf.keras.applications.xception.Xception(weights='imagenet', include_top=False, input_shape=img_dim)
else:    
    mobilenet_model = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=img_dim)

model_array = [vgg16_model, inception_model, resnet_model, xception_model, mobilenet_model]
model_str = ["vgg16_model", "inception_model", "resnet_model", "xception_model", "mobilenet_model"]
conv_model = model_array[ind]

conv_model.trainable = False
conv_model.summary()

model = Sequential()

model.add(conv_model)
model.add(Flatten())
model.add(Dense(256, activation='relu')) #512,256
model.add(Dense(256, activation='relu')) #512,256
model.add(Dense(16, activation='softmax'))



model.compile(loss='binary_crossentropy', #categorical_crossentropy,binary_crossentropy
              optimizer=Adam(learning_rate=1e-3), # lr=0.001
              metrics=['acc']) #categorical_accuracy, acc

model.summary()

train_data = 'dataset/train'
validation_data = 'dataset/validation'
test_data = 'dataset/test'


train_datagen = ImageDataGenerator(
      rescale=1./255, 
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
      )

train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=(224, 224),
        batch_size=16,
        )


validation_datagen = ImageDataGenerator(
        rescale=1./255
        )

validation_generator = validation_datagen.flow_from_directory(
        validation_data,
        target_size=(224, 224),
        batch_size=16,
        )


history = model.fit(
          train_generator,
          validation_data=validation_generator,
          epochs=50, 
          )

model.save('classification_model.h5')


test_datagen = ImageDataGenerator(
        rescale=1./255
        )

test_generator = test_datagen.flow_from_directory(
        test_data,
        target_size=(224, 224),
        batch_size=32,
        )


results = model.evaluate(test_generator) 
print("test loss, test acc:", results)
