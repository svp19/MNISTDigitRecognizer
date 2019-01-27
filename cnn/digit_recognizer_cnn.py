import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import keras.backend as K
K.set_image_dim_ordering('th') #images -> [channels,rows,columns]

np.random.seed(7)

#Loading MNIST dataset
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#Reshape -> [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0],1,28,28).astype('float32')
X_test = X_test.reshape(X_test.shape[0],1,28,28).astype('float32')

#Normalize
X_train = X_train/255
X_test = X_test/255

#One Hot Encoder
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print("Num_classes : ",num_classes)


def cnn_model():
	model = Sequential()
	model.add(Conv2D(32,(5,5),input_shape=(1,28,28),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128,activation='relu'))
	model.add(Dense(num_classes,activation='softmax'))

	model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
	return model

model = cnn_model()
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=200,verbose=1)

#Evaluation
scores = model.evaluate(X_test,y_test,verbose = 0)
print("CNN Error : %.2f%%" %(100-scores[1]*100))

from keras.models import model_from_json
import os

cnn_model_json = model.to_json()
with open("trained_model/model.json", "w")as json_file:
	json_file.write(cnn_model_json)
model.save_weights("trained_model/model.h5")
print("Successfully Saved Trained CNN Model!")
