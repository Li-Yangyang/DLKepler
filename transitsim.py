# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 09:45:52 2018

@author: yangyang
"""
#from transit_basic import *
import kepio, keputils
import numpy as np
from KOI_simulation_utils import *
import matplotlib.pylab as plt
from transit_basic import *
from kepplot import *

def io(cata_path, columnsnames, newnames):
    renames = dict(zip(columnsnames, newnames))
    paras_frame = kepio.get_property(cata_path, columnsnames)
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

def simtransitpop(cata_path, data_path, n):
    """
    """
    paras_frame = io(cata_path, ['koi_duration', 'koi_srad', 'koi_smass'], \
    ['duration', 'srad', 'smass'])
    duration = paras_frame['duration'].values
    srad = paras_frame['srad'].values
    smass = paras_frame['smass'].values
    i = 0
    while(i<n):
        try:
            import warnings
            warnings.filterwarnings("error")
            pop = ParaSampler(smass, srad, duration)
            warnings.resetwarnings()
            B = LcNoiseSampler(cata_path, data_path)
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
            #bareflux = bareflux[sorted_i]
            #transitflux = transitflux[sorted_i]
            std = np.std(flux)
            print(B.kepid,std)
            TP = TransitProfile(pop.P_pop, t0, phase, flux, pop.duration_pop)
            injection = TP.check_snr()
            if injection == 1.0:
                plt.figure(figsize=(10,8))
                plt.scatter(phase, flux, marker='.', linestyle='None', color = 'black')
                #plt.ylim(0.995, 1.0025)
                plt.title('folded_lc#'+str(i))
                plt.xlabel('phase mod period (days)')
                plt.ylabel('Normalized Brightness')
                plt.savefig("./testsim/"+str(i)+"_addglobalnoise.png")
                plt.close()
                #plt.scatter(phase, bareflux,  marker='.', linestyle='None', color = 'black')
                #plt.scatter(phase, transitflux,  marker='.', linestyle='None', color = 'red')
                #plt.figure(figsize=(10,8))
                #plt.scatter(t, transitflux, marker='.', linestyle='None', color='black')
                #plt.savefig("./testsim/"+str(i)+"_globalnoise_notfold.png")
                #plt.show()
                #plt.close()
                #print(np.shape(phase), np.shape(flux))
                #local view plot:
                profile = TP.transit_profile
                plt.scatter(profile['phase'], profile['flux'], marker='.', linestyle='None', color = 'black')
                plt.title('folded_lc_local_vielw#'+str(i))
                plt.xlabel('phase mod period (days)')
                plt.ylabel('Normalized Brightness')
                plt.savefig("./testsim/"+str(i)+"_addglobalnoise(local_view).png")
                plt.close()
                print('this is the orbital period inserted %s (days)' % pop.P_pop)
                print('this is the duration inserted %s (hrs)' % pop.duration_pop)
                print('this is the semi-major axis %s (AU)' % pop.a_pop)
                print('this is the planet radius %s (stellar radius)' % pop.rprs_pop)
                print('this is the stellar radius %s (sun radius)' % pop.srad_pop)
                print('this is the inclination angle %s' % pop.inc_pop)
                i = i + 1
            else:
                continue
        except RuntimeWarning:
            pass


    #return all_phase, all_norflux

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='transit lightcurve population simulation generator')
    parser.add_argument("-v", "--version",action='version', version='%(prog)s 0.5')
    parser.add_argument('n', help='population number.',\
                        type=int)
    cata_path = '../catalog/cumulative.csv'
    data_path = '/scratch/kepler_data/'
    args = parser.parse_args()
    simtransitpop(cata_path, data_path, n=args.n)
