# -*- coding: utf-8 -*-
"""
Object Recognition Problem Statement Checkpoint - 02
Date of Submission : 10-Feb-2019
"""

import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
#from keras.applications.xception import Xception
#from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Dense,GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Model, load_model
from keras.optimizers import Adam

from sklearn.metrics import classification_report

model = load_model('mobilenet_ooo3_015_10_16_model_2')

#print(model.summary())

model.load_weights('mobilenet_ooo3_015_10_16_2.h5')

# Load Image Data Set Using Keras
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   validation_split=0.15)
    
training_set = train_datagen.flow_from_directory('dataset/training_set_test',
                                                 target_size = (128, 128),
                                                 batch_size = 16,
                                                 class_mode = 'categorical',
                                                 subset = 'training')
    
validation_set = train_datagen.flow_from_directory('dataset/training_set_test',
                                                 target_size = (128, 128),
                                                 batch_size = 16,
                                                 class_mode = 'categorical',
                                                 subset = 'validation')

#model.evaluate_generator(training_set, 49)

#model.evaluate_generator(validation_set, 49)

test_csv = pd.read_csv('dataset/object-recognition-predictions.csv')
prediction_set = test_csv['prediction']

key_set = []
for key in training_set.class_indices.keys():
    key_set.append(key)

error_prediction_count = 0

for i in range(1365):
    img_path = 'dataset/Object_classification_test_data/{0}.jpg'.format(i+1)
    #img_path = 'dataset/Object_classification_test_data'
    test_image = image.load_img(img_path, target_size = (128, 128))
    test_image = image.img_to_array(test_image) /255.
    test_image = np.expand_dims(test_image, axis = 0)
    #result = model.predict_classes(test_image)
    #print(result)
    result = model.predict(test_image)
    #print(result)
    max_prob = np.max(result)
    print(max_prob)
    ind = np.where(result == max_prob)[1][0]
    test_csv['prediction'][i] = key_set[ind]
    prediction_set[i] = key_set[ind]
    if max_prob < 0.9:
        error_prediction_count = error_prediction_count + 1
    '''for key, value in training_set.class_indices.items():
        if ind == value:
            test_csv['prediction'][i] = key
            prediction_set[i] = key'''

export_csv = test_csv.to_csv (r'dataset/Object_M1041921_20190222_02.csv', index = None, header=True)

print (export_csv)

print(classification_report(prediction_set, prediction_set, labels = key_set))


"""
K = 5 # We want the indices of the four largest values
index = result[0].argsort()[-K:][::-1]
index = index.tolist()
index_value = result[0][index]*100
index_key = ["", "", "", "", ""]



for key, value in training_set.class_indices.items():
    if index[0] == value:
        index_key[0] = key
    if index[1] == value:
        index_key[1] = key
    if index[2] == value:
        index_key[2] = key
    if index[3] == value:
        index_key[3] = key
    if index[4] == value:
        index_key[4] = key
        
      
        
probability = [
    {
        "index" : index[0],
        "percent" : round(index_value[0], 2),
        "category" : index_key[0]
    },
    {
        "index" : index[1],
        "percent" : round(index_value[1], 2),
        "category" : index_key[1]
    },
    {
        "index" : index[2],
        "percent" : round(index_value[2], 2),
        "category" : index_key[2]
    },
    {
        "index" : index[3],
        "percent" : round(index_value[3], 2),
        "category" : index_key[3]
    },
    {
        "index" : index[4],
        "percent" : round(index_value[4], 2),
        "category" : index_key[4]
    }
]
    
print(pred_name)
#plt.imshow(test_image[0])
"""
