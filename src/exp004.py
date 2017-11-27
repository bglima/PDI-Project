# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:46:39 2017

@author: tvieira@ic.ufal.br
"""

#%% Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#%% Initialising the CNN
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(256, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#%% Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

#%%
training_set = train_datagen.flow_from_directory('../db/test002/training_set/',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

#%%
test_set = test_datagen.flow_from_directory('../db/test002/test_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

#%% Fit the classifier
classifier.fit_generator(training_set,
                         steps_per_epoch = training_set.n,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = test_set.n)

#score = classifier.evaluate_generator(test_set, 8*40)

#%% Save the classification model architecture and weights
from class2file import class2json
import os
exp = 'exp004'
folder = '../res/' + exp + '/'
if not os.path.exists(folder):
    os.makedirs(folder)

class2json(classifier, folder + exp)

#%% Test some images to check whether the classifier has actual high accuracy
#from class2file import json2class
#iFold = 1
#classifier_filename = '../model/classifier_fold' + str(iFold) + '/' + 'classifier_fold'  + str(iFold)
#classifier = json2class(classifier_filename)
#import numpy as np
#from keras.preprocessing import image
#test_image = image.load_img('../cnn/F1/test/P7/image_dist111.png', 
#                            target_size = (64, 64), grayscale = True)
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#classifier.predict(test_image)





























