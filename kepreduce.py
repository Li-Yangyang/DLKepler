# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 23:34:56 2018

@author: yangyang
"""

from astropy.io import fits
import numpy as np
import kepio, keputils, kepspline
from astropy.stats import sigma_clip

def reduce_lc(instr, lc_path):
    tstart, tstop, bjdref, cadence = kepio.timekeys(instr, lc_path)

    #read lc
    hdu = instr[1]
    time = hdu.data.TIME
    time = time +bjdref -2454900
    flux = hdu.data.PDCSAP_FLUX
    #filter data
    work1 = np.array([time, flux])
    work1 = np.rot90(work1, 3)
    work1 = work1[~np.isnan(work1).any(1)]

    intime = work1[:,1]
    indata = work1[:,0]
    #split lc
    intime, indata = keputils.split(intime, indata, gap_width = 0.75)
    #calculate breaking points
    bkspaces = np.logspace(np.log10(0.5), np.log10(20), num = 20)
    #calculate spline to every data points
    spline = kepspline.choose_kepler_spline(intime, indata, bkspaces, penalty_coeff = 1.0, verbose=False)[0]
    if spline is None:
        raise ValueError("faied to fit spline")


    #flatten the data array
    intime = np.concatenate(intime).ravel()
    indata = np.concatenate(indata).ravel()
    spline = np.concatenate(spline).ravel()
    #do sigma cilp later
    #normalized flux using spline
    nordata = indata/spline
    #sigma clip to remove outliers
    #nordata = sigma_clip(nordata, 3, 5)
    #mask = nordata.mask
    #intime = np.ma.masked_array(intime, mask=mask)
    #compress yo ndarray
    #intime = intime.compressed()
    #nordata = nordata.compressed()

    return intime, nordata

def fetchtseries(instr, lc_path):
    tstart, tstop, bjdref, cadence = kepio.timekeys(instr, lc_path)

    #read lc
    hdu = instr[1]
    time = hdu.data.TIME
    time = time +bjdref -2454900
    flux = hdu.data.PDCSAP_FLUX
    #filter data
    work1 = np.array([time, flux])
    work1 = np.rot90(work1, 3)
    work1 = work1[~np.isnan(work1).any(1)]

    intime = work1[:,1]
    #indata = work1[:,0]
    #split lc
    #intime, indata = keputils.split(intime, indata, gap_width = 0.75)

    #adopt uniform distribution
    #intime = np.concatenate(intime).ravel()
    #print(intime.min, intime.max, intime.size)
    tseries = np.linspace(np.min(intime), np.max(intime), np.shape(intime)[0])

    return tseries
