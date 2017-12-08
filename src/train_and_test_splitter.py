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
from shutil import copyfile

#%%

def get_file_list_from_dir(data_dir):
    all_files = os.listdir(os.path.abspath(data_dir))
    data_files = list(filter(lambda file: file.endswith('.jpg'), all_files))
    return data_files

def randomize_files(file_list):
    shuffle(file_list)
    
def get_train_and_test_sets(file_list, split):
    split_index = int(floor(len(file_list) * split))
    training_set = file_list[0:split_index]
    test_set = file_list[split_index:len(file_list)]
    return training_set, test_set
    
def check_folder(folder_dir):
    if not os.path.exists(folder_dir):    
        print(folder_dir + " didn't exist. Creating now...")
        os.makedirs( folder_dir )
    else:
        print(folder_dir + " exists. Using it...")  
        
def save_results(training_set, test_set, imgs_dir, ann_dir, out_dir):

    train_dir = os.path.join(out_dir,'train_set/') 
    test_dir = os.path.join(out_dir,'test_set/') 
    
    check_folder(train_dir)
    check_folder(test_dir)

    ## Name files in crescent order   
    for img_name in training_set:
        im = cv2.imread(imgs_dir+ '/' + img_name)
        cv2.imwrite(train_dir + img_name, im)      
        # Copy xmls to train_dir     
        xmlPath = os.path.splitext(ann_dir + '/' + img_name)[0] + '.xml'
        newXmlPath = train_dir + os.path.splitext(img_name)[0]  + '.xml'
        if( not os.path.exists(xmlPath) ):    # Copy XML if exists
            print("[ERR] " + xmlPath + " not found...")
            continue       
        copyfile(xmlPath, newXmlPath )
        
      
    for img_name in test_set:
        im = cv2.imread(imgs_dir + '/' + img_name)
        cv2.imwrite(test_dir + img_name, im)        
        # Copy xmls to test_dir    
        xmlPath = os.path.splitext(ann_dir + '/' + img_name)[0] + '.xml'
        newXmlPath = test_dir + os.path.splitext(img_name)[0]  + '.xml'
        if( not os.path.exists(xmlPath) ):    # Copy XML if exists
            print("[ERR] " + xmlPath + " not found...")
            continue       
        copyfile(xmlPath, newXmlPath )
    return
    
#%%

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images_dir", required=True,
	help="path to dir that includes images")
ap.add_argument("-a", "--annotations_dir", required=True,
	help="path to dir that includes annotations")
ap.add_argument("-o", "--output_dir", required=True,
	help="path to dir that will contain the train_set and test_set folders")
ap.add_argument("-s", "--split", type=float, default=0.7,
	help="Percentage to split data. Default is 0.7 for training and 0.3 for test") 
args = vars(ap.parse_args())

#%%
args = {}
args["images_dir"] = '../db/training003/tire/images'
args["annotations_dir"] = '../db/training003/tire/annotations'
args["output_dir"] = '../db/training003/'
args["split"] = 0.7

imgs_dir = args["images_dir"]
ann_dir = args["annotations_dir"]
out_dir = args["output_dir"]
split = args["split"]

imgs = get_file_list_from_dir( imgs_dir )   
randomize_files( imgs )
train, test = get_train_and_test_sets( imgs, split)
save_results(train, test, imgs_dir, ann_dir, out_dir)

print("Ended split successfully")
