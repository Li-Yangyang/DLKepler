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
import h5py

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
                    
def save_to_hdf(para, lc, label, filename=None):
    if filename is None:
        filename = os.path.join('./result/','simpopset.h5')
    #mode:  Read/write if exists, create otherwise (default)                               
    f = h5py.File(filename,'a')
    grp = f.create_group(str(label))
    names = ['parameters','data']
    formats = ['S16','f8']
    dtype = dict(names = names, formats=formats)
    para_array = np.array(list(para.items()), dtype=dtype)                               
    lc_array = lc.values
    grp.create_dataset('parameters', data=para_array)
    grp.create_dataset('lc', data=lc_array)
                                    
    return f
                                    
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
    
    paras_frame = io(self.catalog, ['koi_duration', 'koi_srad', 'koi_smass'], \
            ['duration', 'srad', 'smass'])
    duration = paras_frame['duration'].values
    srad = paras_frame['srad'].values
    smass = paras_frame['smass'].values
    i = 0
    while(i<args.n):
        try:
            para, lc = simulate_one(smass, srad, duration, catalog, data_path)
            if(para!=None and lc!=None):     
                i = i + 1
                #save to hdf
                sf = save_to_hdf(para, lc, args.n)
                sf.close()                    
        except RuntimeWarning:
            pass

    #simtransitpop(catalog, data_path, n=args.n)
    end = time.time()
    print(end - start)
