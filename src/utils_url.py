#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 18:24:50 2017
@author: tvieira

Modified by BG
"""

def getFileFromURL (url, filename):

    import sys

    if sys.version_info[0] == 2:
        """If Python version is 2.7"""
        import urllib
        url_obj = urllib.URLopener()
        try:
            url_obj.retrieve(url, filename)
        except:
            print( 'Error retrieving url ' + url )
            return False
        return True
    
    if sys.version_info[0] == 3:
        """If Python version is 3"""
        import urllib.request
        try:
            urllib.request.urlretrieve(url, filename)
        except:
            print( 'Error retrieving url ' + url )
            return False
        return True