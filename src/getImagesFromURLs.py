# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:24:18 2017
@author: tvieira
"""

#%% Import libraries
import urllib

#%% Read the list of images from a .txt file
def readListFromFile( filename ):
    """Return a list containing all lines from text file"""
    try:
        with open( filename ) as f:
            lines = f.readlines()
            return lines
    except:
        #e = sys.exc_info()[0]
        print( 'Error reading file ' + filename )

#%% Get file from url
def downloadFileFromURL (url, file_name):
    url_obj = urllib.URLopener()
    try:
        url_obj.retrieve(url, file_name)
    except:
        #e = sys.exc_info()[0]
        print( 'Error reading url ' + url )

#%% Get images for specific category
# Categories can be:
# aedes
# culex
# tire
# bucket
# trash
def getImgsFromCategory (category, images_to_download):
    import os
    from utils_url import getFileFromURL
    image_list = '../db/list_' + category + '.txt'
    output_dir = '../db/' + category + '/'
    outfile_prefix = category + '_'
    lines = readListFromFile( image_list )
    # Create folder if it doesn't exist
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    # Download all images from specific URLs
    images_downloaded = 0
    for i in range(1, len(lines)):
        if images_downloaded == images_to_download: # Break if reaches number of images
            break
        
        url_path = lines[i]
        url_path = url_path.replace('/r','').replace('\n', '')
        print('\n')
        print(str(i) + ' ' + url_path)
        file_name, file_extension = os.path.splitext(url_path)
        output_filename = output_dir + outfile_prefix + str(i) + file_extension
        getOk = getFileFromURL(url_path, output_filename)
 
        if (not getOk):   # Doens't count if could not retrieve URL
            continue

        # If it get to here, it's allright!
        images_downloaded += 1
        
#%%
def cleanDabataseDirectory(directory):
    import os
    from os.path import isfile, join
    from os import listdir
    import cv2
    
    # Create an array with all valid images path into dir
    validTypes = ('.jpg', '.jpeg', '.gif', '.png', '.tga')
    imgDirPath = '../db/' + directory
    if not os.path.exists(imgDirPath):
        print('Directory "'+directory+'" not found...')
        return
    # Starting cleaning itself
    print('Starting cleaning in "'+imgDirPath+'"...')
    for f in listdir(imgDirPath):
        path = join(imgDirPath, f)
        if not path.lower().endswith(validTypes): 
            os.remove(path)
            print('[WF] Removing "'+path+'" - Wrong format.')
            continue
        img = cv2.imread(path)
        if img is None: 
            print('[FC] Removing "'+path+'" - File corrupted.')
            os.remove(path)
            continue
    print('Directory clean.')

     
#%%
def chooseImages(directory, numberOfImgsToChoose):
    import os
    from os.path import isfile, join
    from os import listdir
    import cv2
    
    imgDirPath = '../db/' + directory
    if not os.path.exists(imgDirPath):
        print('Directory "'+imgDirPath+'" not found...')
        return
    else:
        print('Chosing images in '+directory+'.')
        
    cv2.namedWindow('img')
    imgPath = []
    for f in listdir(imgDirPath):
        path = join(imgDirPath, f)
        imgPath.append(path)
    
    imgsInPath = len(imgPath)
    print('There are '+str(imgsInPath)+' in path')

    imgIndex = 0
    chosenImgs = 0
    img = cv2.imread(imgPath[0])
    cv2.imshow('img', img)
 
    while(True):
        # Check if key was pressed
        key = cv2.waitKey(20)
        if (key == -1) : continue
        # Update image index
        elif ( key == ord('q') ) : 
            break
        elif ( key == ord('y') ) : 
            imgIndex += 1
            chosenImgs += 1
            if( chosenImgs % 10 == 0 ):
                print( str(chosenImgs) + ' images were chosen.')
        elif ( key == ord('n') ):
            os.remove(imgPath[imgIndex])
            imgIndex += 1
            
        
        # If images ended, break
        if( imgIndex == imgsInPath ):
            print('No more images.')
            break

        # If all images were chosen, remove the rest
        if (chosenImgs == numberOfImgsToChoose ):
            while( imgIndex < imgsInPath ):
                print('Removing '+imgPath[imgIndex]+'...')
                os.remove(imgPath[imgIndex])
                imgIndex += 1
            break
        else:            
            img = cv2.imread(imgPath[imgIndex])
            cv2.imshow('img', img)
    
    # Destroy all windows after quiting program    
    cv2.destroyAllWindows()        
    print(str(chosenImgs)+' images were chosen.')

#%%
def copyAnnotations(directory):
    import os
    from os.path import isfile, join
    from os import listdir
    from shutil import copyfile
    
    imgDirPath = '../db/' + directory
    annDirPath = '../db/Annotation/' + directory + '/'
    
    if not os.path.exists(imgDirPath):
        print('Directory "'+imgDirPath+'" not found...')
        return
    elif not os.path.exists(annDirPath):
        print('Annotations directory "Annotation/'+directory+'" not found.')
        return

    notFound = []
    for f in listdir(imgDirPath):
        annotationName = os.path.splitext(f)[0] + '.xml'
        annotationPath = annDirPath + annotationName
        
        if os.path.exists(annotationPath):
            print('Copying annotation for '+f+'...')
            copyfile(annotationPath, imgDirPath + '/' + annotationName)
        else:
            notFound.append(annotationPath)
            
    if (len(notFound)):
        print('Some annotations were not found:')
        for ann in notFound:
            print(ann)
    print('Finished copying annotations')