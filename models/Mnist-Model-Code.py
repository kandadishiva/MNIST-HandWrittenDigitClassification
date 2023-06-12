# Feed Forward Neural Network
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

#loading the dataset
(x_Train,y_Train),(x_Test,y_Test)=mnist.load_data()
test=x_Test

#Processing the data

print(x_Train.shape)
x_Train=x_Train.reshape(x_Train.shape[0],28*28)
x_Test=x_Test.reshape(x_Test.shape[0],28*28)

print(x_Train.shape)

y_Train=to_categorical(y_Train)
y_Test=to_categorical(y_Test)

#normalization
x_Train=x_Train.astype("float")
x_Test=x_Test.astype("float")

x_Train=x_Train/255
x_Test=x_Test/255

model=Sequential()
model.add(Dense(512,activation="relu",input_shape=(28*28,),name="layer1"))
model.add(Dense(10,activation="softmax",name="layer2"))

model.summary()

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(x_Train,y_Train,epochs=10)

loss,accuracy=model.evaluate(x_Test,y_Test)

print("Loss ",loss)
print("Accuracy ",accuracy)

#Predection of the image

# Get the test image
test_image = test[0]
plt.imshow(test_image, cmap='gray')
plt.show()

# Reshape and normalize the test image
test_image = test_image.reshape((1, 28 * 28))
test_image = test_image.astype('float32') / 255

# Make prediction
prediction = model.predict(test_image)
predicted_label = np.argmax(prediction[0])

print("Predicted label:", predicted_label)

# Convolutional Neural Network
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 128
num_classes = 10
epochs = 10

model = Sequential()
model.add(Conv2D (32,kernel_size=(5,5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense (128, activation='relu'))
model.add(Dropout (0.3))
model.add(Dense (64, activation='relu'))
model.add(Dropout (0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("The model has successfully trained")

model.save('mnist.h5')
print("Saving the model as mnist.h5")

#Predection of the image
import matplotlib.image as img
# Get the test image


test_image = test[30]
plt.imshow(test_image, cmap='gray')
plt.show()

print(test_image.shape)
# Reshape and normalize the test image
test_image = test_image.reshape((1, 28,28,1))
print(test_image.shape)
test_image = test_image.astype('float32') / 255

# Make prediction
prediction = model.predict(test_image)
predicted_label = np.argmax(prediction[0])

print("Predicted label:", predicted_label)