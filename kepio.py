# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 09:30:11 2018

@author: yangyang
"""

#from astropy.io import fits
#import numpy as np
#import os
#import matplotlib.pyplot as plt
#import glob
import pandas as pd
#import kepstat
#import math
#import kepler_utils
#import kepler_spline
#from astropy.stats import sigma_clipped_stats

def timekeys(instr, filename):
    """read time keywords"""
    tstart = 0.0
    tstop = 0.0
    cadence = 0.0

    # BJDREFI
    try:
        bjdrefi = instr[1].header['BJDREFI']
    except:
        bjdrefi = 0.0

    # BJDREFF
    try:
        bjdreff = instr[1].header['BJDREFF']
    except:
        bjdreff = 0.0
    bjdref = bjdrefi + bjdreff

    # TSTART
    try:
        tstart = instr[1].header['TSTART']
    except:
        try:
            tstart = instr[1].header['STARTBJD'] + 2.4e6
        except:
            try:
                tstart = instr[0].header['LC_START'] + 2400000.5
            except:
                try:
                    tstart = instr[1].header['LC_START'] + 2400000.5
                except:
                    errmsg = ('ERROR -- KEPIO.TIMEKEYS: Cannot find TSTART, '
                              'STARTBJD or LC_START in ' + filename)
                    print(errmsg)
                    #kepmsg.err(logfile, errmsg, verbose)
    tstart += bjdref

    # TSTOP
    try:
        tstop = instr[1].header['TSTOP']
    except:
        try:
            tstop = instr[1].header['ENDBJD'] + 2.4e6
        except:
            try:
                tstop = instr[0].header['LC_END'] + 2400000.5
            except:
                try:
                    tstop = instr[1].header['LC_END'] + 2400000.5
                except:
                    errmsg = ('ERROR -- KEPIO.TIMEKEYS: Cannot find TSTOP, '
                              'STOPBJD or LC_STOP in ' + filename)
                    print(errmsg)
    tstop += bjdref

    # OBSMODE
    cadence = 1.0
    try:
        obsmode = instr[0].header['OBSMODE']
    except:
        try:
            obsmode = instr[1].header['DATATYPE']
        except:
            errmsg = ('ERROR -- KEPIO.TIMEKEYS: cannot find keyword OBSMODE '
                      'or DATATYPE in ' + filename)
            print(errmsg)
    if 'short' in obsmode:
        cadence = 54.1782
    elif 'long' in obsmode:
        cadence = 1625.35

    return tstart, tstop, bjdref, cadence

def get_id(catalog):
    """
    get kepler id from keplerstellar catalog(200038) or kepler koi(8214)
    """
    #catalog_frame= pd.read_csv(cata_path,skiprows=67)
    #drop = catalog_frame.drop_duplicates(subset='kepid') delete this cuz consider multiple planets sys
    kepid = catalog['kepid'].values

    return kepid

def get_property(catalog, select_cols):
    #catalog_frame = pd.read_csv(cata_path, skiprows=67)
    return catalog[select_cols]

def pathfinder(k_id, data_path, quarter):
    """
    find relative light curve data
    args: k_id: kepid from kepler stellar catalog
          data_path: the path in which light curves are
          quarter: target quarter directory
    """
    #file path here:
    #data_path_dir = './'
    if len(str(k_id)) == 6:
        file_dir=data_path+'kepler_Q'+str(quarter)+'/kplr000'+str(int(k_id))+'*.fits'
    elif len(str(k_id)) == 7:
        file_dir=data_path+'kepler_Q'+str(quarter)+'/kplr00'+str(int(k_id))+'*.fits'
    else:
        file_dir=data_path+'kepler_Q'+str(quarter)+'/kplr0'+str(int(k_id))+'*.fits'
    #quarter = 1
    #timescale = 6.5
    return file_dir
