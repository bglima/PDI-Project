# This script was taken from:
# https://github.com/datitran/raccoon_dataset/

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    dataset = 'training007'
    
    for directory in ['train_set', 'test_set']:
        image_path = '../db/' + dataset + '/{}'.format(directory)
        xml_path = '../db/' + dataset + '/' + 'csv'
        
        xml_df = xml_to_csv(image_path)
        if (not os.path.exists( xml_path ) ):
            os.mkdir( xml_path )
        xml_df.to_csv( xml_path + '/{}_labels.csv'.format(directory), index=None)
        print('Successfully converted xml to csv.')

#%%

main()