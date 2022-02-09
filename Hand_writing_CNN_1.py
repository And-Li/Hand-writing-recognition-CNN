from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras import utils
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

print('x_train:', x_train.shape, type(x_train))
print('x_test:', x_test.shape)
print()
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)

x_train = x_train.reshape(128, -1)
print(x_train.shape)


model = Sequential()
model.add(Dense(100, input_dim=x_train.shape[1],activation='relu'))

model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',  # used for output of more than 2 classes ('sparce_CE' takes in class labels in 'discrete numbers' view, not in one-hot-encoding)
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
model.fit(x_train,
          y_train,
          validation_split=0.2,
          shuffle=True,
          batch_size=128,
          epochs=15,
          verbose=1)

