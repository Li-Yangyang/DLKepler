import kepio,kepstat
from astropy.io import fits
import kepreduce
from transit_basic import *

instr = fits.open('/scratch/kepler_data/kepler_Q16/kplr008123937-2013098041711_llc.fits')
filename = '/scratch/kepler_data/kepler_Q16/kplr008123937-2013098041711_llc.fits'
print(kepio.timekeys(instr, filename))

cata_path = '../catalog/cumulative.csv'
data_path = '/scratch/kepler_data/'

B = LcNoiseSampler(cata_path, data_path)
print(B.kepid)
import numpy as np
kepid = np.random.choice(kepio.get_id(cata_path))

filedir = kepio.pathfinder(kepid, data_path, '*')
filenames = glob.glob(filedir)
all_time = []
all_flux = []
for i in range(len(filenames)):
    #read into cadence
    instr = fits.open(filenames[i])
    tstart, tstop, bjdref, cadence = kepio.timekeys(instr, filenames[i])
    #reduce lc
    intime, nordata = kepreduce.reduce_lc(instr, filenames[i])
    #calculte runing stddev
    stddev = kepstat.running_frac_std(intime, nordata, 6.5 / 24) * 1.0e6
    #cdpp
    cdpp = stddev / math.sqrt(6.5 * 3600.0 / cadence)
    # filter cdpp
    for i in range(len(cdpp)):
        if cdpp[i] > np.median(cdpp) * 10.0:
            cdpp[i] = cdpp[i - 1]

    #calculte RMS cdpp
    rms = kepstat.rms(cdpp, np.zeros(len(stddev)))
    rs = np.random.RandomState(seed=13)
    flux1 = np.zeros(nordata.size)+ np.median(nordata)
    errors1 = rms*1e-06*np.ones_like(nordata)
    #add gaussian noise
    flux1 += errors1*rs.randn(len(nordata))
    all_time.append(intime)
    all_flux.append(flux1)
    instr.close()
t = np.concatenate(all_time)
simflux = np.concatenate(all_flux)
print(t.shape, simflux.shape)
