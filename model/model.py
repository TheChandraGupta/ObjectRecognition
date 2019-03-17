"""
Object Recognition Problem Statement Checkpoint - 01
Date of Submission : 10-Feb-2019
"""

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

import pandas as pd
import numpy as np
from keras import backend as K
import keras
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import load_model

class MyModel:
    
    def __init__(self):
        K.clear_session()
		
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True,
                                       validation_split=0.1)
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
        
        self.model = load_model('mobilenet_ooo3_015_10_16_model_2')
        self.model.load_weights('mobilenet_ooo3_015_10_16_2.h5')
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        self.class_indices = validation_set.class_indices
        self.model.summary()
		

    def predict_image(self, img_path):
        base_app_path = 'uploads/'
        img_path = base_app_path + img_path
        test_image = image.load_img(img_path, target_size = (128, 128))
        test_image = image.img_to_array(test_image) /255.
        test_image = np.expand_dims(test_image, axis = 0)
        with self.graph.as_default():
            result = self.model.predict(test_image)
        #print(result)
        
        #max_prob = np.max(result)
        #print(max_prob)
        #for key, value in self.class_indices.items():
        #    if np.where(result == max_prob)[1][0] == value:
        #        pred_name = key
                
        K = 5 # We want the indices of the four largest values
        index = result[0].argsort()[-K:][::-1]
        index = index.tolist()
        index_value = result[0][index]*100
        index_value = index_value.tolist()
        index_key = ["", "", "", "", ""]
        
        for key, value in self.class_indices.items():
            if index[0] == value:
                index_key[0] = key.upper()
            if index[1] == value:
                index_key[1] = key.upper()
            if index[2] == value:
                index_key[2] = key.upper()
            if index[3] == value:
                index_key[3] = key.upper()
            if index[4] == value:
                index_key[4] = key.upper()
        
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
        
        #plt.imshow(test_image[0])	
        print(probability)
        return probability
