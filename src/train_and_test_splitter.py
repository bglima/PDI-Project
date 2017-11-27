# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:37:25 2017

@author: brunolima
"""


import cv2
import os
from random import shuffle
from math import floor
import argparse

#%%

def get_file_list_from_dir(data_dir):
    all_files = os.listdir(os.path.abspath(data_dir))
    data_files = list(filter(lambda file: file.endswith('.jpg'), all_files))
    return data_files

def randomize_files(file_list):
    shuffle(file_list)
    
def get_training_and_testing_sets(file_list, split):
    split_index = int(floor(len(file_list) * split))
    training_set = file_list[0:split_index]
    test_set = file_list[split_index:len(file_list)]
    return training_set, test_set
    
def save_results(training_set, test_set, data_dir):

    train_dir = os.path.join(data_dir,'training_set/') 
    test_dir = os.path.join(data_dir,'test_set/') 
    
    if not os.path.exists(train_dir):    
        os.makedirs( train_dir )
    else:
        print("Training dir already exists. Delete folder if generating again...\n")
        return
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)     
    else:
        print("Training dir already exists. Delete folder if generating again...\n")
        return
    ## Name files in crescent order   
    for img_name in training_set:
        im = cv2.imread(data_dir + '/' + img_name)
        cv2.imwrite(train_dir + img_name, im)        
    for img_name in test_set:
        im = cv2.imread(data_dir + '/' + img_name)
        cv2.imwrite(test_dir + img_name, im)        
    return
    
#%%

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path to dir that include images, that will contain train and test folders")
ap.add_argument("-s", "--split", type=float, default=0.7,
	help="Percentage to split data. Default is 0.7 for training and 0.3 for test") 
args = vars(ap.parse_args())

imgs_dir = args["path"]
split = args["split"]
imgs = get_file_list_from_dir( imgs_dir )    
randomize_files( imgs )
train, test = get_training_and_testing_sets( imgs, split )
save_results(train, test, imgs_dir)
    
