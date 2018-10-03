# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 09:45:52 2018

@author: yangyang
"""
#from transit_basic import *
import kepio, keputils
import numpy as np
import pandas as pd
from KOI_simulation_utils import *
import matplotlib.pylab as plt
from transit_basic import *
from kepplot import *

def io(catalog, columnsnames, newnames):
    renames = dict(zip(columnsnames, newnames))
    paras_frame = kepio.get_property(catalog, columnsnames)
    paras_frame = paras_frame.rename(index=str, columns=renames)
    paras_frame = paras_frame.dropna()

    return paras_frame

class TransitProfile(object):
    """
    """

    def __init__(self, period, t0, phase_array, f, duration):
        self.period = period
        self.t0 = t0
        self.phase_array = phase_array
        self.f = f
        self.duration = duration

        #From Yinan's code kepler_simu_utils.py

        lv = local_view(self.phase_array, self.f, self.period, self.duration)

        t_min= self.duration*-3.0
        t_max= self.duration*3.0

        time_array=np.linspace(t_min, t_max, lv.size)

        data_lc = dict()
        data_lc['phase'] = time_array
        data_lc['flux'] = lv
        self.transit_profile = data_lc

    def check_snr(self):

        time_array=self.transit_profile['phase']#*24.0
        #start to measure the SNR around the signal.
        index_snr = np.argwhere((time_array > self.duration*1.0) | (time_array < self.duration*-1.0))
        lv = self.transit_profile['flux']
        mean_snr, median_snr, std_snr = sigma_clipped_stats(lv[index_snr])

        index_profile = np.argwhere((time_array <= self.duration*1.0) & (time_array >= self.duration*-1.0))

        if lv[index_profile].min() <= median_snr - 3*std_snr:

            return 1.0
        else:
            return 0.0

def simulate_one(smass, srad, duration, catalog, data_path):
    import warnings
    warnings.filterwarnings("error")
    pop = ParaSampler(smass, srad, duration)
    warnings.resetwarnings()
    B = LcNoiseSampler(catalog, data_path)
    t, bareflux = B.generator(6.5)
    t0 = np.random.choice(t)
    transitflux = batman_planet(t0, pop.P_pop, pop.rprs_pop,pop.ars_pop,\
            pop.inc_pop, 0, 90, t)
    final_flux = bareflux*transitflux
    time_array = t
    flux = final_flux
    phase = keputils.phase_fold_time(time_array, pop.P_pop, t0)
    sorted_i = np.argsort(phase)
    phase = phase[sorted_i]
    flux = flux[sorted_i]
    TP = TransitProfile(pop.P_pop, t0, phase, flux, pop.duration_pop)
    injection = TP.check_snr()
    if injection == 1.0:
        profile = TP.transit_profile
        para = {'t0':t0, 'Period':pop.P_pop, 'rprs':pop.rprs_pop, 'ars':pop.ars_pop,\
                'inclination':pop.inc_pop, 'duration':pop.duraiton_pop, 'prekid':B.kepid, 'std': np.std(flux)}
        lc = pd.DataFrame(np.array([profile['phase'], profile['flux']), columns=['phase', 'flux'])
        return para, lc
    else:
        return None, None
                    
class Simtransitpop(object):
    """
    """
    def __ini__(self, catalog, data_path, n):
        self.catalog = catalog
        self.data_path = data_path
        self.n = n
    
        paras_frame = io(self.catalog, ['koi_duration', 'koi_srad', 'koi_smass'], \
            ['duration', 'srad', 'smass'])
        duration = paras_frame['duration'].values
        srad = paras_frame['srad'].values
        smass = paras_frame['smass'].values
        i = 0
        while(i<self.n):
            try:
                para, lc = simulate_one(smass, srad, duration, self.catalog, self.data_path)
                if(para!=None and lc!=None):     
                    i = i + 1
                    #save to hdf
            except RuntimeWarning:
                pass


    #return all_phase, all_norflux

if __name__ == '__main__':
    import argparse
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description='transit lightcurve population simulation generator')
    parser.add_argument("-v", "--version",action='version', version='%(prog)s 0.5')
    parser.add_argument('n', help='population number.',\
                        type=int)
    cata_path = '../catalog/cumulative+noise.csv'
    data_path = '/scratch/kepler_data/'
    args = parser.parse_args()
    catalog= pd.read_csv(cata_path, skiprows=67)
    simtransitpop(catalog, data_path, n=args.n)
    end = time.time()
    print(end - start)
