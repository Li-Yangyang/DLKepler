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

def io(cata_path, columnsnames, newnames):
    renames = dict(zip(columnsnames, newnames))
    paras_frame = kepio.get_property(cata_path, columnsnames)
    paras_frame = paras_frame.rename(index=str, columns=renames)
    paras_frame = paras_frame.dropna()

    return paras_frame

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
            print(std)
            if 3*std>(1-np.min(flux)):
                plt.figure(figsize=(10,8))
                plt.scatter(phase, flux, marker='.', linestyle='None', color = 'black')
                #plt.ylim(0.995, 1.0025)
                plt.title('folded_lc#'+str(i))
                plt.xlabel('phase mod period (days)')
                plt.ylabel('Normalized Brightness')
                plt.savefig("./testsim/"+str(i)+"_globalnoise.png")
                plt.close()
                #plt.scatter(phase, bareflux,  marker='.', linestyle='None', color = 'black')
                #plt.scatter(phase, transitflux,  marker='.', linestyle='None', color = 'red')
                plt.figure(figsize=(10,8))
                plt.scatter(t, transitflux, marker='.', linestyle='None', color='black')
                plt.savefig("./testsim/"+str(i)+"_globalnoise_notfold.png")
                #plt.show()
                plt.close()
                print(np.shape(phase), np.shape(flux))
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
