# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:33:53 2018

@author: brunolima
"""

import os
import cv2
import numpy as np
import imutils as iu

INPUT_PATHS = ['../db/training004/flowerpot/' ]
            #'../db/training004/tire/'   
            #'../db/training004/flowerpot/',
            #'../db/training004/bottle/']    
AUGMENT_OPTIONS = [ 'ROTATION', 'SHEAR' ]
VALID_TYPES = ('.jpg', '.gif', '.png', '.tga')
TARGET_SIZE = (300, 300)


def process( aug_type, img_name, img, out_dir ):   
    # ROTATION
    if aug_type == 'ROTATION':
        # 3 Rotations of 90 deg
        angles = [90, 180, 270]
        for angle in angles:           
            img_out = iu.rotate(img, angle)                        
            img_out_name = out_dir + os.path.splitext(img_name)[0] + 'rot' + str(angle) + '.jpg'
            print('Writing %s ...' % (img_out_name) ) 
            cv2.imwrite(img_out_name, img_out)
            
    # SHEAR
    elif aug_type == 'SHEAR':
        # Shear factor definitions
        shear_factors = [0.3, -0.3]      
        for shear_factor in shear_factors:           
            # Horizontal            
            shear_matrix_h = np.array([[1,shear_factor, 0], [0, 1, 0] ])
            img_out = cv2.warpAffine(img, shear_matrix_h, TARGET_SIZE)
            img_out_name = out_dir + os.path.splitext(img_name)[0] + 'shearH' + str(shear_factor) + '.jpg'
            print('Writing %s ...' % (img_out_name) ) 
            cv2.imwrite(img_out_name, img_out)
            
            # Vertical
            shear_matrix_v = np.array([[1, 0, 0], [shear_factor, 1, 0]])
            img_out = cv2.warpAffine(img, shear_matrix_v, TARGET_SIZE)
            img_out_name = out_dir + os.path.splitext(img_name)[0] + 'shearV' + str(shear_factor) + '.jpg'
            print('Writing %s ...' % (img_out_name) ) 
            cv2.imwrite(img_out_name, img_out)
    else:
        raise Exception('%s augmentation command not known.')

    print('All right!')
    
def augment_datasets():
    # Iterate over folders
    for input_dir in INPUT_PATHS:
        img_dir = os.path.join(input_dir, 'images/')            
        ann_dir = os.path.join(input_dir, 'annotations/')
        out_dir = os.path.join(input_dir, 'augmented/')
        
        # Check if folders exist
        if not os.path.exists(img_dir):
            raise Exception('%s dir not found.'% (img_dir) )
        if not os.path.exists(ann_dir):
            raise Exception('%s dir not found.'% (ann_dir) )            

        # Create output dir if not exists
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Load names from img directory            
        for img_name in os.listdir(img_dir):
            # Load image from path
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            
            # Check if its a valid image
            if (img is not None) and (img_path.lower().endswith(VALID_TYPES)):
                # First of all, resize the image and write to output folder
                img = cv2.resize(img, TARGET_SIZE, interpolation = cv2.INTER_CUBIC)
                cv2.imwrite(out_dir+img_name, img)
            
                # Augmentation option from each image goes here
                for arg in AUGMENT_OPTIONS: process(arg, img_name, img, out_dir)

def main():
    augment_datasets()
    
if __name__ == '__main__':
    main()
    