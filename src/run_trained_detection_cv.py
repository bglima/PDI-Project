# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 22:51:56 2017

@author: Bruno
"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from os import listdir 
from os.path import isfile, join

import cv2

#%%

MODEL_NAME = 'training005/output'            # What model is going to be used
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'                # Frozen detection graph
PATH_TO_LABELS = os.path.join('training005', 'object-detection.pbtxt')  # Class labels
NUM_CLASSES = 3                             # Num of classes into your model
PATH_TO_TEST_IMAGES_DIR = 'test_images'     # Folder containing test images
MIN_CONFIDENCE = 0.7

if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")  
from utils import label_map_util
from utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#%%
# Loading a (frozen) Tensorflow model into memory

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
	
# Loading a label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#%%

# Create an array with all valid images path into dir
valid_types = ('.jpg', '.jpg', '.gif', '.png', '.tga')
image_path = []
print("[INFO] checking and loading images...")
for file_name in listdir(PATH_TO_TEST_IMAGES_DIR):
    file_path = join(PATH_TO_TEST_IMAGES_DIR, file_name)
    if not file_path.lower().endswith(valid_types): continue
    image = cv2.imread(file_path)
    if image is None: continue
    image_path.append(file_path)

if ( len(image_path) == 0 ):
    print("[ERR] No images were found at " + PATH_TO_TEST_IMAGES_DIR )
    raise
    
print("[INFO] images loaded successfully...")

# Load first image
cv2.namedWindow('input', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)
image_index = 0
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

COLORS = np.random.uniform(0, 255, size=(len(categories), 3))

#%%

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite dinput and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    print("[INFO] Started TF Session. Press any key to continue...")
    while True: 
            # Check if key was pressed
            key = cv2.waitKey(50)
            if (key == -1) : continue
            
            # Update image index
            elif ( key == ord('q') ): 
                cv2.destroyAllWindows()
                break
            elif ( key == ord('a') and image_index > 0 ) : 
                image_index -= 1
            elif ( key == ord('d') and image_index < len(image_path)-1 ):
                image_index += 1

            # Open the image with updated index
            image_np = cv2.imread( image_path[image_index] ) 
            image_np_out = image_np.copy()   # Will contain results
            (h, w) = image_np.shape[0:2]
           
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            
            # Creating boxes around detections
            index = 0
            while( scores[0][index] > MIN_CONFIDENCE ):
                box = boxes[0][index] * np.array([h, w, h, w])
                (y_start, x_start, y_end, x_end) = box.astype("int")
                print("[BOX] ", box.astype("int") )
                # Print prediction info
                class_idx = classes[0][index].astype("int") 
                label = "{}: {:.2f}%".format( categories[ class_idx - 1 ]['name'], scores[0][index] * 100)
                print("[INFO] {}".format(label))
                
                # Display box
                cv2.rectangle(image_np_out, (x_start, y_start), (x_end, y_end), COLORS[class_idx - 1], 2)
                index += 1
                
            # Show input and output images
            cv2.imshow('input', image_np)
            cv2.imshow('output', image_np_out)
            
# Destroy all windows after quiting program  
cv2.destroyAllWindows()