#import data
from keras.datasets import mnist
import matplotlib.pyplot as plt 

(X_train,y_train),(X_test,y_test) = mnist.load_data() 
'''
#See data

plt.subplot(221)
plt.imshow(X_train[0], cmap = plt.get_cmap('gray'))

plt.subplot(222)
plt.imshow(X_train[1], cmap = plt.get_cmap('gray'))

plt.subplot(223)
plt.imshow(X_train[2], cmap = plt.get_cmap('gray'))

plt.subplot(224)
plt.imshow(X_train[3], cmap = plt.get_cmap('gray'))
plt.show()
'''

#simple multilayer peceptron model
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

#random
np.random.seed(7) 

#print (X_train.shape)

#Flatten 28*28 images into row vectors for keras
numpixels = X_train.shape[1] * X_train.shape[2] 
X_train = X_train.reshape(X_train.shape[0],numpixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0],numpixels).astype('float32')

#Normalization
X_train = X_train/255 
X_test = X_test/255 

#One Hot Encoder for 1-9 output
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]

#Neural Net Model
model = Sequential() 
model.add(Dense(numpixels , input_dim = numpixels , kernel_initializer = 'normal' , activation = 'relu' ))
model.add(Dense(num_classes,kernel_initializer='normal',activation = 'softmax'))
#logarithmic loss(cross entropy)
model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

#Fit model to training data
model.fit(X_train,y_train, validation_data=(X_test,y_test),epochs=10,batch_size=250,verbose=2)

#Final Results

scores =  model.evaluate(X_test,y_test,verbose=0)
print("Score : %.2f%%" % (scores[1]*100))