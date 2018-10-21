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
import urllib

def reframe(catalog, columnsnames, newnames):
    renames = dict(zip(columnsnames, newnames))
    paras_frame = kepio.get_property(catalog, columnsnames)
    paras_frame = paras_frame.rename(index=str, columns=renames)
    paras_frame = paras_frame.dropna()

    return paras_frame

class TransitProfile(object):
    """
    """

    def __init__(self, period, t0, phase_array, f, model_f, duration, depth):
        self.period = period
        self.t0 = t0
        self.phase_array = phase_array
        self.f = f
        self.model_f = model_f
        self.duration = duration
        self.depth = depth

        #From Yinan's code kepler_simu_utils.py

        lv = keputils.local_view(self.phase_array, self.f, self.period, self.duration/24.0)

        t_min= self.duration*-3.0/24.0
        t_max= self.duration*3.0/24.0

        time_array=np.linspace(t_min, t_max, lv.size)

        data_lc = dict()
        data_lc['phase'] = time_array
        data_lc['flux'] = lv
        self.transit_profile = data_lc

    def check_snr(self):
        intra_idx = np.argwhere((self.phase_array < self.duration*1.0/24.0) & (self.phase_array > self.duration*-1.0/24.0))
        area_model = np.sum(abs(1-self.model_f[intra_idx]))
        area_real = np.sum(abs(1-self.f[intra_idx]))
        rms = np.sqrt(np.sum((self.model_f[intra_idx]-self.f[intra_idx])**2.0))
        area_diff = abs(area_model-area_real)
        #plt.scatter(self.phase_array[intra_idx], self.f[intra_idx], marker='.', color='black', alpha='0.2')
        #plt.scatter(self.phase_array[intra_idx], self.model_f[intra_idx], marker='.', color='green', alpha='0.2')
        #plt.scatter(self.phase_array[intra_idx], np.ones(len(self.model_f[intra_idx]))*(self.depth), marker='.', color='orange', alpha='0.2')
        #plt.show()
        #plt.close()
        #plt.scatter(self.phase_array, self.model_f, marker='.', color='orange', alpha='0.2')
        #plt.show()
        #step = max(phase_array[intra_idx])-min(phase_array[intra_idx])
        print(area_model, area_diff, self.period, self.duration, self.depth, rms)
        #if (area_model/rms)>3.0:
        if (area_model/area_diff)>0.5:
          #plt.scatter(self.phase_array[intra_idx], self.f[intra_idx], marker='.', color='black', alpha='0.2')
          #plt.scatter(self.phase_array[intra_idx], self.model_f[intra_idx], marker='.', color='green', alpha='0.2')
          #plt.show()
          time_array=self.transit_profile['phase']#*24.0
          ##start to measure the SNR around the signal.
          index_snr = np.argwhere((self.phase_array > self.duration/24.0*1.0) | (self.phase_array < self.duration/24.0*-1.0))
          index_snr_lv = np.argwhere((self.transit_profile['phase'] > self.duration/24.0*1.0) | (self.transit_profile['phase'] < self.duration/24.0*-1.0))
          #lv = self.transit_profile['flux']
          mean_snr, median_snr, std_snr = sigma_clipped_stats(self.f[index_snr])
          mean_snr_lv, median_snr_lv, std_snr_lv = sigma_clipped_stats(self.transit_profile['flux'][index_snr_lv])

          #index_profile = np.argwhere((time_array <= self.duration*1.0) & (time_array >= self.duration*-1.0))

          #if 1-3*std_snr >= 1-self.depth:

          return 1.0, std_snr, std_snr_lv
        else:
          return 0.0, None, None

def simulate_one(smass, srad, duration, catalog, data_path, injection):
    import warnings
    warnings.filterwarnings("error")
    pop = ParaSampler(smass, srad, duration)
    warnings.resetwarnings()
    B = LcNoiseSampler(catalog, data_path, pop.rprs_pop)
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
         cube = np.transpose(p.map(B.generateone, task, 4))
         t = cube[0]
         simflux = cube[1]
         #print(np.shape(cube))
    t = np.concatenate(t)
    bareflux = np.concatenate(simflux)

    t0 = np.random.choice(t)
    if injection == 1.0:

        transitflux = batman_planet(t0, pop.P_pop, pop.rprs_pop,pop.ars_pop,\
                pop.inc_pop, 0, 90, t)
        #t0, per, rp, a, inc, ecc, w, t
        final_flux = bareflux*transitflux
        time_array = t
        flux = final_flux
        phase = keputils.phase_fold_time(time_array, pop.P_pop, t0)
        sorted_i = np.argsort(phase)
        phase = phase[sorted_i]
        flux = flux[sorted_i]
        transitflux = transitflux[sorted_i]
        TP = TransitProfile(pop.P_pop, t0, phase, flux, transitflux, pop.duration_pop, 1-pop.rprs_pop**2)
        cksnr = TP.check_snr()[0]
        if cksnr == 1.0:
            profile = TP.transit_profile
            para = {'t0':t0, 'Period':pop.P_pop, 'rprs':pop.rprs_pop, 'ars':pop.ars_pop,\
                    'inclination':pop.inc_pop, 'duration':pop.duration_pop, 'prekid':B.kepid,\
                     'std_nobining': TP.check_snr()[1], 'std_bining': TP.check_snr()[2],'flag': 1.0}
            lc = np.array([profile['phase'], profile['flux']])

            #plt.scatter(phase, flux, color='black', marker='.', alpha=0.2, label=str(1.0))
            #plt.scatter(phase, np.ones(len(phase))*(1-pop.rprs_pop**2.0), marker='.', color='g', alpha=0.2, label='depth')
            #plt.show()
            #print(para)
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
        TP = TransitProfile(pop.P_pop, t0, phase, flux, pop.duration_pop, 1-pop.rprs_pop**2.0)
        profile = TP.transit_profile
        para = {'t0':t0, 'Period':pop.P_pop, 'rprs':pop.rprs_pop, 'ars':pop.ars_pop,\
                'inclination':pop.inc_pop, 'duration':pop.duration_pop, 'prekid':B.kepid,\
                 'std': np.std(flux), 'flag':0.0}
        lc = np.array([profile['phase'], profile['flux']])

        return para, lc, injection

def save_to_hdf(para, lc, seq, filename=None):
    if filename is None:
        filename = os.path.join('./result/','popset.h5')
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
    parser = argparse.ArgumentParser(description='transit lightcurve population simulation generator(snr check update)')
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
            if(args.flag==1.0):
                if(cksnr==1.0):
                    i = i + 1
                    #save to hdf
                    sf = save_to_hdf(para, lc, i, './result/simpopset8000.h5')
                    sf.close()
            else:
                i = i + 1
                sf = save_to_hdf(para,lc,i, './resul/zeropopset8000.h5')
                sf.close()
        except (RuntimeWarning, TimeoutError, urllib.error.URLError) as e:
            pass

    #simtransitpop(catalog, data_path, n=args.n)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('./log/'+str(args.flag)+str(args.n)+'.txt', 'w+') as f:
        f.write(s.getvalue())
