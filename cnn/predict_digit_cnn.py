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
from keras.models import model_from_json
import helpers as hp

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

#LOAD CNN from JSON

json_file = open('trained_model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("trained_model/model.h5")
print("Loading CNN...Success!")

'''
loaded_model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')
scores = loaded_model.evaluate(X_test,y_test,verbose=0)
print("Error : %.2f%%" % (100-scores[1]*100))
'''

#Testing Prediction on image
image_to_predict = 'images/testdigit_1.png'
hp.plot_and_predict_sm(image_to_predict, loaded_model)
# Alternative : hp.plot_and_predict_cv2(image_to_predict, loaded_model)
