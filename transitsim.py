# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 09:45:52 2018

@author: yangyang
"""
#from transit_basic import *
import os
import kepio, keputils
import numpy as np
import pandas as pd
from KOI_simulation_utils import *
import matplotlib.pylab as plt
from transit_basic import *
import h5py
import kplr
from multiprocessing import Pool

def reframe(catalog, columnsnames, newnames):
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

        lv = keputils.local_view(self.phase_array, self.f, self.period, self.duration)

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

def simulate_one(smass, srad, duration, catalog, data_path, injection):
    import warnings
    warnings.filterwarnings("error")
    pop = ParaSampler(smass, srad, duration)
    warnings.resetwarnings()
    B = LcNoiseSampler(catalog, data_path)
    client = kplr.API('/media/yinanzhao/Backup_7/kepdata')
    #star = client.star(kepid)
    lcs = client.light_curves(B.kepid,short_cadence=False)
    #filenames.sort()
    #datepat = re.compile(r'Q+\d+')
    #index = 0
    #for i in range(len(filenames)):
    #    index = datepat.findall(filenames[i])[0][1:]
    #    rms = self.catalog.loc[lambda df: df.kepid == self.kepid, 'Q'+index+'rms'].values[0]
    #    #read into cadence
    #    instr = fits.open(filenames[i])
    #    #read into time series
    #    intime = kepreduce.fetchtseries(instr, filenames[i])
    #    #simulate
    #    rs = np.random.RandomState(seed=13)
    #    flux1 = np.zeros(intime.size)+ 1.0
    #    errors1 = rms*np.ones_like(intime)#use global std
    #    #errors1 = cdpp*1e-06*np.ones_like(nordata)*math.sqrt(timescale * 3600.0 / cadence)#correct from cdpp to std
    #    #add gaussian noise
    #    flux1 += errors1*rs.randn(len(intime))
    #    all_time.append(intime)
    #    all_flux.append(flux1)
    #    instr.close()
    with Pool(5) as p:
         task = [ lc for lc in lcs]
         #print(task)
         #prod_lc=partial(generateone, kepid=self.kepid, catalog=self.catalog)
         t, simflux = np.transpose(p.map(B.generateone, task, 4))
         #print(t)
    t = np.concatenate(t)
    bareflux = np.concatenate(simflux)

    t0 = np.random.choice(t)
    if injection == 1.0:

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
        cksnr = TP.check_snr()
        if cksnr == 1.0:
            profile = TP.transit_profile
            para = {'t0':t0, 'Period':pop.P_pop, 'rprs':pop.rprs_pop, 'ars':pop.ars_pop,\
                    'inclination':pop.inc_pop, 'duration':pop.duration_pop, 'prekid':B.kepid,\
                     'std': np.std(flux), 'flag': 1.0}
            lc = np.array([profile['phase'], profile['flux']])

            return para, lc, cksnr
        else:
            return None, None, cksnr
    else:
        final_flux = bareflux
        time_array = t
        flux = final_flux
        phase = keputils.phase_fold_time(time_array, pop.P_pop, t0)
        sorted_i = np.argsort(phase)
        phase = phase[sorted_i]
        flux = flux[sorted_i]
        TP = TransitProfile(pop.P_pop, t0, phase, flux, pop.duration_pop)
        profile = TP.transit_profile
        para = {'t0':t0, 'Period':pop.P_pop, 'rprs':pop.rprs_pop, 'ars':pop.ars_pop,\
                'inclination':pop.inc_pop, 'duration':pop.duration_pop, 'prekid':B.kepid,\
                 'std': np.std(flux), 'flag':0.0}
        lc = np.array([profile['phase'], profile['flux']])

        return para, lc

def save_to_hdf(para, lc, seq, filename=None):
    if filename is None:
        filename = os.path.join('./result/','simpopsettest.h5')
    #mode:  Read/write if exists, create otherwise (default)
    f = h5py.File(filename,'a')
    grp = f.create_group(str(seq))
    names = ['parameters','data']
    formats = ['S16','f8']
    dtype = dict(names = names, formats=formats)
    para_array = np.array(list(para.items()), dtype=dtype)
    lc_array = lc
    grp.create_dataset('parameters', data=para_array)
    grp.create_dataset('lc', data=lc_array)

    return f

if __name__ == '__main__':
    import argparse
    import time
    import cProfile
    import pstats
    import io
    pr = cProfile.Profile()
    pr.enable()
    parser = argparse.ArgumentParser(description='transit lightcurve population simulation generator')
    parser.add_argument("-v", "--version",action='version', version='%(prog)s 0.5')
    parser.add_argument('n', help='population number.',\
                        type=int)
    parser.add_argument('flag', help='injection or not', type=int)
    cata_path = '../catalog/cumulative+noise.csv'
    data_path = '/scratch/kepler_data/'
    args = parser.parse_args()
    catalog= pd.read_csv(cata_path, skiprows=67)

    paras_frame = reframe(catalog, ['koi_duration', 'koi_srad', 'koi_smass'], \
            ['duration', 'srad', 'smass'])
    duration = paras_frame['duration'].values
    srad = paras_frame['srad'].values
    smass = paras_frame['smass'].values
    i = 0
    while(i<args.n):
        try:
            para, lc, cksnr= simulate_one(smass, srad, duration, catalog, data_path, args.flag)
            #print(flag)
            if(cksnr==1.0):
                i = i + 1
                #save to hdf
                sf = save_to_hdf(para, lc, i)
                sf.close()
        except RuntimeWarning:
            pass

    #simtransitpop(catalog, data_path, n=args.n)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('test5.txt', 'w+') as f:
        f.write(s.getvalue())
