import os
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


# mute tensorflow build warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

print('Loading model...')
model = load_model('IQA_model.h5')


IMG_WIDTH = IMG_HEIGHT = 150

img_path = input('Image name: ')

while img_path != '':
	img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
	img = np.asarray(img)
	img = img/255
	img = np.expand_dims(img, axis=0)

	print('Predicting...')
	results = model.predict(img)
	
	print(results)

	img_path = input('Image name: ')
