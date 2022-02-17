# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 09:58:02 2022

@author: YALÇINKAYA
"""

from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


from matplotlib import pyplot as plt
import numpy as np


nameArray = ["Taze Elma","Taze Muz","Taze Portakal","Çürük Elma","Çürük Muz","Çürük Portakal"]

model = load_model('classification_model.h5')
#elma.jpeg
#portakal.jpg
#muz.jpeg
img = load_img("mix.jpeg", target_size=(224,224))

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




