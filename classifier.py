from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from keras.preprocessing.image import image
from keras.preprocessing.image import img_to_array
from tensorflow import keras
import sys

fn = sys.argv[1]

file_path = os.path.join('/dog-classifier/input/', fn)

if not os.path.exists(file_path):
    print('File does not exists')
    exit()

model = keras.models.load_model('/dog-classifier/dog_breed_model')

img = image.load_img(file_path, target_size = (224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img/255

res = model.predict(img)

with open('classes.json') as json_file:
        labels = json.load(json_file)
labels = dict((v,k) for k,v in labels.items())

max_results = 3

pred_idx = res[0].argsort()[-max_results:][::-1]

result = {}

for i in range(max_results):
    result[labels[pred_idx[i]]] = res[0][[pred_idx[i]]][0]

print(result)




