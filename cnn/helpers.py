import cv2
import scipy.misc as sm
import numpy as np
import matplotlib.pyplot as plt

def plot_and_predict_sm(image_to_predict, loaded_model):
	x_orig = sm.imread(image_to_predict,mode='L')
	#compute a bit-wise inversion so black becomes white and vice versa
	x = np.invert(x_orig)
	#make it the right size
	x = sm.imresize(x,(28,28))
	plt.imshow(x_orig, cmap = plt.get_cmap('gray'))
	#convert to a 4D tensor to feed into our model
	x = x.reshape(1,1,28,28)
	x = x.astype('float32')
	x /= 255

	#perform the prediction

	out = loaded_model.predict(x)
	result = "Predicted Value :" + str(np.argmax(out))
	print(result)
	plt.title(result)
	plt.savefig('result.png')
	plt.show()

def plot_and_predict_cv2(image_to_predict, loaded_model):
	image_orig = cv2.imread('testdigit2.png',0)
	image = np.invert(image_orig)
	image = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT)
	x = image
	#compute a bit-wise inversion so black becomes white and vice versa
	#x = image
	#x = np.invert(x)

	print(x.shape)
	#convert to a 4D tensor to feed into our model
	x = x.reshape(1,1,28,28)
	x = x.astype('float32')
	x /= 255
	print('Predicting...')
	#perform the prediction

	out = loaded_model.predict(x)
	result = "Predicted Value :" + str(np.argmax(out))
	print(result)

	plt.title(result)
	plt.imshow(image_orig, cmap = plt.get_cmap('gray'))
	plt.savefig('result.png')
	plt.show()