# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:33:53 2018

@author: brunolima
"""

import os
import cv2
from random import random
import numpy as np
import imutils as iu
from lxml import etree
from io import StringIO
import math

INPUT_PATHS = [ 
            '../db/training007/emptybottle/'#, 
            #'../db/training007/tire/', 
            #'../db/training007/flowerpot/' 
            ]
AUGMENT_OPTIONS = [ 'ROTATION' ]
VALID_TYPES = ('.jpg', '.gif', '.png', '.tga')
TARGET_SIZE = (300, 300)
OX, OY = TARGET_SIZE[0] // 2, TARGET_SIZE[1] // 2

# Override original images with TARGET_SIZE images, keeping aspect ratio
# Does not consider annotations
def resize_datasets():
    # Iterate over folders
    for input_dir in INPUT_PATHS:
        img_dir = os.path.join(input_dir, 'images/')            
        
        # Check if folders exist
        if not os.path.exists(img_dir):
            raise Exception('%s dir not found.'% (img_dir) ) 
    
        # Load names from img directory            
        for img_name in os.listdir(img_dir):
            # Load image from path
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
    
             # Read old size
            old_width, old_height = img.shape[1], img.shape[0]
            aspect_ratio = float(old_width) / float(old_height)
            
            # Check if size is already ok
            if old_width == TARGET_SIZE[0] and old_height == TARGET_SIZE[1]:
                print(' %s already satisfies dimension requirements.' % img_name )
                continue
            # Resize max dimension to 300, keeping ratio
            elif old_width > old_height:
                new_width = TARGET_SIZE[0]
                new_height = int( float(new_width) / float(aspect_ratio) )
                
            else:
                new_height = TARGET_SIZE[1]                                
                new_width = int( float(new_height) * float(aspect_ratio) )
            
            # Find borders
            delta_w = TARGET_SIZE[0] - new_width
            delta_h = TARGET_SIZE[1] - new_height
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            
            # Resizing and creating a bordered version
            img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_CUBIC )
            img_w_borders = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
            
            # Saving
            print('Resizing %s and saving' % img_name)
            cv2.imwrite(img_dir+img_name, img_w_borders)
            
# Save new data to annotation
def save_to_ann( ann_tree, ann_path, img_out_path, xmin, ymin, xmax, ymax ):
    # Getting elments from XML
    data_path = ann_tree.getroot().find('path')
    data_xmin = ann_tree.getroot().find('object/bndbox/xmin')
    data_ymin = ann_tree.getroot().find('object/bndbox/ymin')
    data_xmax = ann_tree.getroot().find('object/bndbox/xmax')
    data_ymax = ann_tree.getroot().find('object/bndbox/ymax')

    # Debugging data 
    print('Annotation from %s: %s %s %s %s' % (data_path.text, 
                                               data_xmin.text, 
                                               data_ymin.text, 
                                               data_xmax.text, 
                                               data_ymax.text) )   
    # Editing values from elements
    data_path.text = os.path.abspath(img_out_path)
    data_xmin.text = str(xmin)
    data_ymin.text = str(ymin)
    data_xmax.text = str(xmax)
    data_ymax.text = str(ymax)
    
    # Saving in ann_path specified directory
    ann_tree.write(ann_path)

# Get bounding box from annotation
def get_bb_from_ann( ann_tree ):
    xmin = ann_tree.getroot().find('object/bndbox/xmin').text
    ymin = ann_tree.getroot().find('object/bndbox/ymin').text
    xmax = ann_tree.getroot().find('object/bndbox/xmax').text
    ymax = ann_tree.getroot().find('object/bndbox/ymax').text
    return int(xmin), int(ymin), int(xmax), int(ymax)

# Rotate arround origin counterclockwise, where
#   px, py = point
#   ox, oy = origin
#   angle = radians    
def rotate_point( px, py, ox, oy, angle  ):
    # Defining difference between point and origin, and angles
    dx = px - ox
    dy = py - oy
    c = math.cos( -angle )
    s = math.sin( -angle )
    
    # Origin matrix
    org_mat = np.array([ox, oy]).reshape(2, 1)
    # Offset matrix
    dif_mat = np.array([dx, dy]).reshape(2, 1)
    # Rotation matrix
    rot_mat = np.array([[ c, -s ], [ s, c ]])
    
    # Resulting point
    return org_mat + np.dot(rot_mat, dif_mat)


# Get rotated bounding box
def rotate_bb( xmin, ymin, xmax, ymax, angle ):
    p1x, p1y = rotate_point( xmin, ymin, OX, OY, angle )
    p2x, p2y = rotate_point( xmax, ymax, OX, OY, angle )
    return int(min(p1x, p2x)), int(min(p1y, p2y)), int(max(p1x, p2x)), int(max(p1y, p2y))

# Augmentation process for each iamge
def process( aug_type, img_name, img, ann_tree, out_dir, ann_out_dir ): 
    # Name without extension
    img_raw_name = os.path.splitext(img_name)[0]
    xmin, ymin, xmax, ymax = get_bb_from_ann( ann_tree )
    
    # ROTATION
    if aug_type == 'ROTATION':
        # 3 Rotations of 90 deg
        angles = [90, 180, 270]
        for angle in angles:           
            # Image processing
            img_out = iu.rotate(img, angle)              
            nx_min, ny_min, nx_max, ny_max = rotate_bb( xmin, ymin, xmax, ymax, np.radians(angle) )            
            # New image path
            img_out_name = img_raw_name + 'rot' + str(angle)                   
            img_out_path = out_dir + img_out_name + '.jpg'            
            # New annotation path
            ann_path = ann_out_dir + img_out_name + '.xml'          
            # Saving
            print('Writing %s ...' % (img_out_path) ) 
            print('XML being saved to %s...\n\n' % ann_path )
            cv2.imwrite(img_out_path, img_out)
            save_to_ann(ann_tree, ann_path, img_out_path, nx_min, ny_min, nx_max, ny_max)
            
    # SHEAR
    elif aug_type == 'SHEAR':
        # Shear factor definitions
        shear_factors = [0.3, -0.3]      
        for shear_factor in shear_factors:           
            # Image processing - Horizontal           
            shear_matrix_h = np.array([[1,shear_factor, 0], [0, 1, 0] ])
            img_out = cv2.warpAffine(img, shear_matrix_h, TARGET_SIZE)
            # New image path
            img_out_name = img_raw_name + 'shearH' + str(shear_factor) 
            img_out_path = out_dir + img_out_name + '.jpg'
            # New annotation path
            ann_path = ann_out_dir + img_out_name + '.xml'
            # Saving
            print('Writing %s ...' % (img_out_path) ) 
            print('XML being saved to %s...\n\n' % ann_path )
            cv2.imwrite(img_out_path, img_out)
            save_to_ann(ann_tree, ann_path, img_out_path, 0, 0, 0, 0)
            
            # Image Processing - Vertical
            shear_matrix_v = np.array([[1, 0, 0], [shear_factor, 1, 0]])
            img_out = cv2.warpAffine(img, shear_matrix_v, TARGET_SIZE)
            # New image path
            img_out_name = img_raw_name + 'shearV' + str(shear_factor)
            img_out_path = out_dir + img_out_name  + '.jpg'
            # New annotation path
            ann_path = ann_out_dir + img_out_name + '.xml'
            # Saving
            print('Writing %s ...' % (img_out_path) ) 
            print('XML being saved to %s...\n\n' % ann_path )
            cv2.imwrite(img_out_path, img_out)
            save_to_ann(ann_tree, ann_path, img_out_path, 0, 0, 0, 0)
            
    # SALT AND PEPPER
    elif aug_type == 'NOISE':     
        # Creating noise based on image
        gaussian_noise = img.copy()
        m = (20, 20, 20)
        s = (20, 20, 20)
        cv2.randn(gaussian_noise, m, s)   # Mean 0 and sigma 150
        # Adding noise to original image
        img_out = img + gaussian_noise
        # New image path
        img_out_name = img_raw_name + 'noise'
        img_out_path = out_dir + img_out_name + '.jpg'
        # New annotation path
        ann_path = ann_out_dir + img_out_name + '.xml'
        # Saving
        print('Writing %s ...' % (img_out_path) ) 
        cv2.imwrite(img_out_path, img_out)
        save_to_ann(ann_tree, ann_path, img_out_path, 0, 0, 0, 0)
    else:
        raise Exception('%s augmentation command not known.')
    print('All right!')

# Process each image and store its results in INPUT_PATH/augmented folder.
def augment_datasets():
    # Iterate over folders
    for input_dir in INPUT_PATHS:
        img_dir = os.path.join(input_dir, 'images/')            
        ann_dir = os.path.join(input_dir, 'annotations/')
        out_dir = os.path.join(input_dir, 'augmented/images/')
        ann_out_dir = os.path.join(input_dir, 'augmented/annotations/')
        
        # Check if folders exist
        if not os.path.exists(img_dir):
            raise Exception('%s dir not found.'% (img_dir) )
        if not os.path.exists(ann_dir):
            raise Exception('%s dir not found.'% (ann_dir) )                 

        # Create output dir if not exists
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(ann_out_dir):
            os.makedirs(ann_out_dir)

        # Load names from img directory            
        for img_name in os.listdir(img_dir):
            # Load image from path
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            
            # Annotation for current image
            ann_path = os.path.join(ann_dir, os.path.splitext(img_name)[0] + '.xml' )
            ann_tree = etree.parse(ann_path)
            
            # Check if its a valid image
            if (img is not None) and (img_path.lower().endswith(VALID_TYPES)):
                # Save current image and annotation to output folder
                cv2.imwrite(out_dir+img_name, img)
                ann_tree.write(ann_out_dir + os.path.splitext(img_name)[0] + '.xml')
                
                # Augmentation option from each image goes here
                for arg in AUGMENT_OPTIONS: process(arg, img_name, img, ann_tree, out_dir, ann_out_dir)

def main():
     augment_datasets()
        
if __name__ == '__main__':
    main()
    
    #%%