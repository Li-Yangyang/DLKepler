import numpy as np
import pandas as pd
from astropy.io import fits
import kepio, kepreduce
import glob
import re
from astropy.stats import sigma_clipped_stats

cata_path = '../catalog/cumulative.csv'
data_path = '/scratch/kepler_data/'
catalog_frame= pd.read_csv(cata_path,skiprows=65)

kepid = catalog_frame['kepid'].values

noise_frame = pd.DataFrame()
for i in range(3):
    filedir = kepio.pathfinder(kepid[i], data_path, '*')
    filenames = glob.glob(filedir)
    filenames.sort()
    datepat = re.compile(r'Q+\d+')
    index = []
    for j in range(len(filenames)):
        index.append(datepat.findall(filenames[j])[0][1:])
    d = dict(zip(index, filenames))
    #newfilenames = [d[k] for k in sorted(d.keys(), key=int)]
    qua = sorted(d.keys(), key=int)
    noise_dict = dict()
    for j in range(18):
        if str(j) not in qua:
            noise_dict['Q'+str(j)+'rms'] = 'NaN'
        else:
            instr = fits.open(d[str(j)])
            tstart, tstop, bjdref, cadence = kepio.timekeys(instr, d[str(j)])
            #reduce lc
            intime, nordata = kepreduce.reduce_lc(instr, d[str(j)])
            #do sigma_clip
            mean, median, std = sigma_clipped_stats(nordata, sigma = 3.0, iters = 5)
            noise_dict['Q'+str(j)+'rms'] = std
    temp_frame = pd.DataFrame([noise_dict])
    noise_frame = noise_frame.append(temp_frame, ignore_index=True)
    print(qua)
print(noise_frame)
    
