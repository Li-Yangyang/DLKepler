# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 12:38:23 2018

@author: yangyang
"""

import math
import random
import numpy as np
from scipy import linalg

def running_frac_std(time, flux, wid):
    """calculate running fractional standard deviation across the array flux
       within a window of width wid
    """

    hwid = wid / 2
    runstd = np.zeros(len(flux))
    for i in range(len(time)):
        valsinwid = flux[np.logical_and(time < time[i] + hwid, time > time[i] - hwid)]
        runstd[i] = np.std(valsinwid) / np.mean(valsinwid)

    return np.array(runstd)
    
def rms(array1, array2):
    """root mean square of two arrays"""
    if len(array1) != len(array2):
        message  = ("ERROR -- KEPSTAT.RMS: Arrays have unequal sizes - "
                    "array1 = {0}, array2 = {1}".format(len(array1),
                                                        len(array2)))
        print(message)

    return math.sqrt(np.nanmean((array1 - array2) ** 2))

