# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 19:07:58 2018

@author: yangyang
"""
import kepio
import glob
import kepstat
import numpy as np
from astropy.io import fits
import math
import matplotlib.pylab as plt
import time
from functools import wraps
import keputils
import kepspline
from scipy.stats import sigmaclip
 
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
                (function.func_name, str(t1-t0))
                )
        return result
    return function_timer

def rmsestimation(quarter, timescale):
    """
    do stastitical analysis of rms cdpp(expected snr) of certain quarter
    """
    #read data from keplerstellar catalog and match lightcurve in lc database
    cata_path = '../catalog/cumulative.csv'
    data_path = '/scratch/kepler_data/'
    kepid = kepio.get_id(cata_path)
    all_rms = []
    own_kid = []

    for i,k_id in enumerate(kepid[:100]):
        #print('This is '+str(kepid[i]))
        #lc path here
        file_dir = kepio.pathfinder(k_id, data_path, quarter)
        try: 
            filename = glob.glob(file_dir)
            name = filename[0]
            #open file and read time keys
            instr = fits.open(name)
            tstart, tstop, bjdref, cadence = kepio.timekeys(instr, filename)
            
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
            #normalized flux using spline
            nordata = indata/spline
            #calculte runing stddev
            stddev = kepstat.running_frac_std(intime, nordata, timescale / 24) * 1.0e6
            #cdpp
            cdpp = stddev / math.sqrt(timescale * 3600.0 / cadence)
            # filter cdpp
            for i in range(len(cdpp)):
                if cdpp[i] > np.median(cdpp) * 10.0:
                    cdpp[i] = cdpp[i - 1]
                    
            #calculte RMS cdpp
            rms = kepstat.rms(cdpp, np.zeros(len(stddev)))
            rmscdpp = np.ones((len(cdpp)), dtype='float32') * rms
            if rms>200:
                plt.figure(figsize=(10,8))
                #plt.hist(cdpp, bins =25, color = 'gold', fill = True, edgecolor = 'black', linewidth = 2.0)
                #plt.ylabel('Count')
                #plt.xlabel('CDPP/6.5hrs(ppm)')
                #plt.savefig("./test/KOIQ"+str(quarter)+str(k_id)+"cdpp.png")
                plt.scatter(intime, nordata, marker='.')
                plt.savefig("./test/KOIQ"+str(quarter)+str(k_id)+"display_lc.png")
            # print('%d has RMS %.1fhr CDPP = %d ppm\n' % (k_id,timescale, rms))
            all_rms.append(rms)
            own_kid.append(k_id)
            
        except IndexError:
            pass
        
    c, low, upp = sigmaclip(all_rms, 3, 3)
    plt.figure(figsize=(10,8))
    plt.hist(c, bins = 25, range=(low,upp), fill = True, edgecolor = 'black', linewidth = 2.0)
    plt.ylabel('Count')
    plt.xlabel('RMSCDPP(ppm)')
    #plt.savefig("./result/KOIQ"+str(quarter)+"rms.png")
    result = np.transpose([own_kid, all_rms])
    #np.savetxt("./result/KOIQ"+str(quarter)+"rms.txt",result)
    #print("#Done! Number of stellar in Quarter %s is %d" % (quarter, np.shape(all_rms)[0]))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='rms estimation of a certain quarter')
    parser.add_argument("-v", "--version",action='version', version='%(prog)s 0.5')
    parser.add_argument('quarter', help='quarter no.',\
                        type=int)
    parser.add_argument('timescale', help='window scale for running box',\
                        type=float)
    args = parser.parse_args()
    rmsestimation(quarter=args.quarter, timescale=args.timescale)
