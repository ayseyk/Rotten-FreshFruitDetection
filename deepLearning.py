# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:26:01 2021

@author: YALÇINKAYA
"""
#import tensorflow as tf

from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


from matplotlib import pyplot as plt
import numpy as np




conv_model = MobileNet(weights='imagenet', 
                       include_top=False,
                       input_shape=(224, 224, 3))

conv_model.summary()

conv_model.trainable = True
set_layer_trainable = False

for layer in conv_model.layers:
    if layer.name == 'conv_dw_13':# 'conv_dw_13'ten itibaren eğitir // block5_conv1
        set_layer_trainable = True
    if set_layer_trainable:
        layer.trainable = True
    else:
        layer.trainable = False



model = Sequential()

model.add(conv_model)
model.add(Flatten())
model.add(Dense(256, activation='relu')) #512,256
model.add(Dense(6, activation='softmax'))


model.compile(loss='binary_crossentropy', #categorical_crossentropy,binary_crossentropy
              optimizer=RMSprop(lr=1e-5),
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
        batch_size=32,
        )


validation_datagen = ImageDataGenerator(
        rescale=1./255
        )

validation_generator = validation_datagen.flow_from_directory(
        validation_data,
        target_size=(224, 224),
        batch_size=32,
        )


history = model.fit(
          train_generator,
          steps_per_epoch=28,
          epochs=10, #sadece epoch olabilir.
          validation_data=validation_generator,
          validation_steps=2
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


results = model.evaluate(test_generator,steps=8) # step silinebilir.
print("test loss, test acc:", results)






















nameArray = ["Taze Elma","Taze Muz","Taze Portakal","Çürük Elma","Çürük Muz","Çürük Portakal"]

#model = load_model('classification_model.h5')

img = load_img("apple.jpeg", target_size=(224,224))

img_array = img_to_array(img)
img_batch = np.expand_dims(img_array, 0)  
img_prep = preprocess_input(img_batch)
prediction = model.predict(img_prep)

np.set_printoptions(suppress=True) # e götürdü

print(prediction)


newArray=prediction[0]

max = newArray[0]
index=0

for i in range(newArray.size): #6 yı kapsamaz
    if(max < newArray[i]):
        max = newArray[i]
        index=i

print(index)

print(nameArray[index])
print(newArray[index])

newArray[index] = np.around(newArray[index],2)

if(newArray[index]<=0.70):
    plt.title("Tanımsız Sınıf!",fontsize=20,color='#be2da8',fontweight='bold')
else:
    plt.title(nameArray[index],fontsize=20,color='#be2da8',fontweight='bold')


plt.text(x=10,y=210,s="Doğruluk Oranı: ",fontdict={"color":"r","fontsize":13, "style":"italic"})
plt.text(x=100,y=210,s=newArray[index],fontdict={"color":"r","fontsize":13, "style":"italic"})
plt.axis('off')
plt.imshow(img)




plt.figure(figsize=(14,3))

plt.subplot(1, 2, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'],'--')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train_acc', 'val_acc'], loc='upper left')
plt.grid(alpha=0.4)
# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'],'--')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper right')
plt.grid(alpha=0.4)
plt.show()










