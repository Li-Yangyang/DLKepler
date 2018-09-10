# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:35:18 2018

@author: yangyang
"""

import matplotlib.pylab as plt
from astropy.io import fits
import os
import time
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

def file_name():
    """
    read from current file directory.
    """
    L=[]
    lc_file_dir = os.path.join(os.getcwd())
    for root, dirs, files in os.walk(lc_file_dir):  
        for file in files:
            try:
                if file.split(os.extsep,2)[1] == 'fits':  
                    L.append(os.path.join(root, file))
                
            except IndexError:
                continue
    return L  
    
def read_data(lc):
    """
    read light curve data into pandas frame.
    """
    l = fits.open(lc)
    t = l[1].data['TIME']
    t = t.byteswap().newbyteorder()
    pdcflux = l[1].data['PDCSAP_FLUX']
    pdcflux = pdcflux.byteswap().newbyteorder()
    pdcerr = l[1].data['PDCSAP_FLUX_ERR']
    pdcerr = pdcerr.byteswap().newbyteorder()
    lcframe = pd.DataFrame({'time':t, 'pdcflux':pdcflux, 'pdcflux_err':pdcerr})
    
    return lcframe
    
def proccess_lc(lc_dataframe):
    """
    1.NANs time points padding with zeros
    2.replace NANs flux with zeros 
    3.normolize light curve
    """
    lc_dataframe = lc_dataframe[pd.notnull(lc_dataframe['time'])]
    median = lc_dataframe['pdcflux'].dropna().median()
    lc_dataframe['pdcflux'] = lc_dataframe['pdcflux'].fillna(median)
    lc_dataframe['pdcflux'] = (lc_dataframe['pdcflux'] - lc_dataframe['pdcflux'].mean())/lc_dataframe['pdcflux'].std()
    
    return lc_dataframe    

def pca(lcset):
    flux_cube = []
    for i in range(len(lcset)):
        lcframe = read_data(lcset[i])
        pro_lcframe = proccess_lc(lc_dataframe=lcframe)
        flux_cube.append(pro_lcframe['pdcflux'].values)
        
    return flux_cube
        
        

    
        