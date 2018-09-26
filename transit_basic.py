# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 15:28:26 2018

@author: yangyang
"""

import numpy as np
import kepio, kepreduce, kepstat
import matplotlib.pylab as plt
import glob
from astropy.io import fits
from astropy import units as u
from astropy.constants import G
from astropy import constants as const
from astropy.stats import sigma_clipped_stats
import math

#cata_path = '../../../../catalog/Kepler/cumulative.csv'
#columns = ['koi_ror', 'koi_duration', 'koi_sma', 'koi_period', 'koi_srad', 'koi_smass']
#paras = ['rprs', 'duration', 'a', 'P', 'srad', 'smass']
#rename = dict(zip(columns, paras))
#paras_frame = kepio.get_property(cata_path, columns)
#paras_frame = paras_frame.rename(index=str, columns=rename)
#paras_frame = paras_frame.dropna()

def incl(rprs, a, srad, duration, P):
    a = 215.0537*a
    ars = a/srad
    duration = duration/24.0
    frac1 = (1+rprs)**2.0 - ars**2.0
    frac2 = ars**2.0*(np.sin(np.pi*duration/P)**2.0-1)
    incl = np.arcsin(np.sqrt(frac1/frac2))*180/np.pi
    return incl

class ParaSampler(object):
    """
    P is uniform distribution from 10 days to 100days
    rprs is uniform distribution from 0 to 0.1
    duration is from 97% percentile range
    Rs is from 99% percentile range
    Ms is from uniform distribution from min to max
    a is calculted from (P/365/M)^(1/3)
    inc is from cos-1(sqrt(((1+rprs)^2-(aoR*sin(piT/P))^2)/aoR^2))
    """

    def __init__(self, smass, srad, duration):
        self.smass = smass
        self.srad = srad
        self.duration = duration

        P_pop = np.random.uniform(10,100, 1)[0]
        smass_pop = np.random.uniform(np.min(self.smass), np.max(self.smass), 1)[0]
        a_pop = (smass_pop)**(1.0/3)*(P_pop/365.0)**(2.0/3)
        rprs_pop = np.random.uniform(0.001,0.1, 1)[0]
        duration_pop = np.random.uniform(np.min(self.duration), np.percentile(self.duration, 97), 1)[0]
        srad_pop = np.random.uniform(np.min(self.srad), np.percentile(self.srad, 99), 1)[0]
        inc_pop = incl(rprs_pop, a_pop, srad_pop, duration_pop, P_pop)

        self.rprs_pop = rprs_pop
        self.duration_pop = duration_pop
        self.a_pop = a_pop
        self.P_pop = P_pop
        self.inc_pop = inc_pop
        self.srad_pop = srad_pop
        self.ars_pop = 15.0537*self.a_pop/self.srad_pop

class LcNoiseSampler(object):
    """
    """
    def __init__(self, cata_path, data_path):
        self.cata_path = cata_path
        self.data_path = data_path
        self.kepid = np.random.choice(kepio.get_id(self.cata_path))

    def generator(self, timescale):
        filedir = kepio.pathfinder(self.kepid, self.data_path, '*')
        filenames = glob.glob(filedir)
        all_time = []
        all_flux = []
        for i in range(len(filenames)):
            #read into cadence
            instr = fits.open(filenames[i])
            tstart, tstop, bjdref, cadence = kepio.timekeys(instr, filenames[i])
            #reduce lc
            intime, nordata = kepreduce.reduce_lc(instr, filenames[i])
            #do sigma_clip
            mean, median, std = sigma_clipped_stats(nordata, sigma = 3.0, iters = 5)
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
            rs = np.random.RandomState(seed=13)
            flux1 = np.zeros(nordata.size)+ median
            errors1 = std*np.ones_like(nordata)#use global std
            #errors1 = cdpp*1e-06*np.ones_like(nordata)*math.sqrt(timescale * 3600.0 / cadence)#correct from cdpp to std
            #add gaussian noise
            flux1 += errors1*rs.randn(len(nordata))
            all_time.append(intime)
            all_flux.append(flux1)
            instr.close()
        t = np.concatenate(all_time)
        simflux = np.concatenate(all_flux)

        return t, simflux
    #i = 0
    #paras_pop = []
    #while(i<10000):
    #    P_pop = np.random.uniform(10,100, 1)
    #    smass_pop = np.random.uniform(np.min(smass), np.max(smass), 1)
    #    a = ((P_pop/365.0)*smass_pop)**(1.0/3.0)
    #    rprs_pop = np.random.uniform(0,0.1, 1)
    #    duration_pop = np.random.uniform(np.min(duration), np.percentile(duration, 97), 1)
    #    srad_pop = np.random.uniform(np.min(srad), np.percentile(srad, 99), 1)
    #    try:
    #        import warnings
    #        warnings.filterwarnings("error")
    #        inc = incl(rprs_pop, a, srad_pop, duration_pop, P_pop)
    #        paras_pop.append([P_pop[0], a[0], rprs_pop[0], inc[0]])
    #        i = i + 1
    #    except RuntimeWarning:
    #        pass

    #paras_pop = np.array(paras_pop)
    #return paras_pop



#duration = paras_frame['duration'].values
#srad = paras_frame['srad'].values
#smass = paras_frame['smass'].values

#pop = sampler(duration, srad, smass)

#plt.hist(pop[:,3],bins=30)
