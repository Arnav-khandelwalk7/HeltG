# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 00:29:33 2020

@author: saigu
"""


from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,Input,Lambda
from keras.models import Model,Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from glob import glob

image_size = [128,128]

train_path = 'data/train'
test_path = 'data/test'

res = ResNet50(include_top=False,weights="imagenet",input_shape=image_size+[3])

for layer in res.layers:
    layer.trainable = False
    
x = Flatten()(res.output)
last_btw = Dense(512,activation='relu')(x)
last = Dense(1,activation='sigmoid')(last_btw)

model = Model(inputs = res.input,outputs=last)
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(128,128),
        batch_size=32,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(128,128),
        batch_size=32,
        class_mode='binary')

model.fit_generator(train_generator,
                        validation_data=validation_generator,
                        epochs=10,
                        steps_per_epoch=len(train_generator),
                        validation_steps=len(validation_generator))





import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='train_loss')
plt.plot(r.history['val_loss'],label='validation_loss')
plt.legend()
plt.show()
plt.savefig('LossValLoss')

plt.plot(r.history['accuracy'],label='train_accuracy')
plt.plot(r.history['val_accuracy'],label='validation_accuracy')
plt.legend()
plt.show()
plt.savefig('Accuracy')

model.save('resnet.h5')










