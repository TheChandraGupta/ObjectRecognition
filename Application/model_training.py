# -*- coding: utf-8 -*-
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

training_set = None

def create_model(img_shape=(299, 299, 3), n_classes=50,
                   load_pretrained=False, freeze_layers_from='base_model'):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None

    # Get base model
    #base_model = Xception(include_top=False, weights=weights,
                       #input_tensor=None, input_shape=img_shape)
    
    #base_model = ResNet50(include_top=False, weights=weights,
                       #input_tensor=None, input_shape=img_shape)
    
    base_model = MobileNet(include_top=False, weights=weights,
                       input_tensor=None, input_shape=img_shape)

    # Add final layers
    x = base_model.output
    x = Dense(n_classes, activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(n_classes, activation='softmax')(x)

    # This is the model we will train
    model = Model(input=base_model.input, output=predictions)
    
    # Model Summary
    # print(model.summary())
    
    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True

    print(model.summary())
    return model
	
def train_model(train_set_path, val_set_path, validation_split = 0.2,
                   batch_size = 32, class_mode = 'categorical', horizontal_flip = False,
                   vertical_flip = False, rotation_range = None, target_size = (299, 299),
                   model = None, epochs = 1, learning_rate = 0.0001, loss = 'categorical_crossentropy',
				   n_classes=50):
    
    # Load Image Data Set Using Keras
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = horizontal_flip,
                                   vertical_flip = vertical_flip,
                                   rotation_range=rotation_range,
                                   validation_split=validation_split)
    
    training_set = train_datagen.flow_from_directory(train_set_path,
                                                 target_size = target_size,
                                                 batch_size = batch_size,
                                                 class_mode = class_mode,
                                                 subset = 'training')
    
    validation_set = train_datagen.flow_from_directory(val_set_path,
                                                 target_size = target_size,
                                                 batch_size = batch_size,
                                                 class_mode = class_mode,
                                                 subset = 'validation')
        
    # Compile Model
    # opt_rms = keras.optimizers.rmsprop(lr=learning_rate,decay=1e-6)
    adam_optimizer = Adam(lr = learning_rate)
    model.compile(optimizer = adam_optimizer, loss = loss, metrics = ['accuracy'])
    
    model.fit_generator(training_set,
                        steps_per_epoch = training_set.samples,
                        epochs = epochs,
                        validation_data = validation_set,
                        validation_steps = validation_set.samples)
        
    model.evaluate_generator(training_set, n_classes)
    model.evaluate_generator(validation_set, n_classes)
    
    return model
	
def model_initialization():
	# Declare Constants
	img_shape = (128, 128, 3)
	target_size = (128, 128)
	n_classes = 49
	epochs = 1
	train_set_path = 'dataset/training_set_test' #'101_ObjectCategories'
	val_set_path = train_set_path
	validation_split = 0.50
	batch_size = 1
	horizontal_flip = True
	rotation_range = None
	learning_rate = 0.0003
	loss = 'categorical_crossentropy'

	# Build Model
	model = create_model(img_shape = img_shape,  n_classes = n_classes, load_pretrained = True)

    #model = load_model('mobilenet_ooo3_015_10_16_model_2')
	#model.load_weights('mobilenet_ooo3_015_10_16_2.h5')

	# Train the Model
	model = train_model(train_set_path = train_set_path, val_set_path = val_set_path, validation_split = validation_split,
						   batch_size = batch_size, horizontal_flip = horizontal_flip, rotation_range = rotation_range,
						   model = model, epochs = epochs, target_size = target_size, learning_rate = learning_rate,
                           loss = loss, n_classes = n_classes)

	model.save_weights('weight_01.h5')
	model.save('model_01')
	json_string = model.to_json()
	f = open('json_model.json', 'w+')
	f.write(json_string)
	f.close()
	
def load_trained_model(img_path):
	base_app_path = 'uploads/'
	model = load_model('mobilenet_model')
	#print(model.summary())
	model.load_weights('mobilenet_weights.h5')
	img_path = base_app_path + img_path
	test_image = image.load_img(img_path, target_size = (128, 128))
	test_image = image.img_to_array(test_image) /255.
	test_image = np.expand_dims(test_image, axis = 0)
	result = model.predict(test_image)
	print(result)

	max_prob = np.max(result)
	for key, value in training_set.class_indices.items():
		if np.where(result == max_prob)[1][0] == value:
			pred_name = key
	
	#plt.imshow(test_image[0])	
	print(pred_name)
	return pred_name
	

model_initialization()

