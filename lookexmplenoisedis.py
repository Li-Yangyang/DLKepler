import numpy as np
import pandas as pd
from astropy.io import fits
import kepio, kepreduce
import glob
import re
from astropy.stats import sigma_clipped_stats
import matplotlib.pylab as plt

cata_path = '../catalog/cumulative.csv'
data_path = '/scratch/kepler_data/'
catalog_frame= pd.read_csv(cata_path,skiprows=65)

kepid = ['3749978', '10232693', '5531576', '8229048', '5380812', '9823457', '9210823', '3656322']
no = ['2', '4', '11', '13', '14', '15', '17', '18', '19']

noise_frame = pd.DataFrame()
for i in range(len(kepid)):
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
    noise_dict['kepid'] = kepid[i]
    for j in range(18):
        if str(j) not in qua:
            noise_dict['Q'+str(j)+'rms'] = float('NaN')
        else:
            instr = fits.open(d[str(j)])
            tstart, tstop, bjdref, cadence = kepio.timekeys(instr, d[str(j)])
            #reduce lc
            intime, nordata = kepreduce.reduce_lc(instr, d[str(j)])
            #do sigma_clip
            mean, median, std = sigma_clipped_stats(nordata, sigma = 3.0, iters = 5)
            noise_dict['Q'+str(j)+'rms'] = std
        plt.scatter(j, noise_dict['Q'+str(j)+'rms']*1e06, color='black')
    plt.title('kepid:'+kepid[i]+'_no:'+no[i])
    plt.xlabel('Quarter')
    plt.ylabel('rms(ppm)')
    plt.savefig('./result/indivnoise/'+no[i]+'.png')
    plt.close()
    #temp_frame = pd.DataFrame([noise_dict])
    #noise_frame = noise_frame.append(temp_frame, ignore_index=True)

#noise_frame.to_csv('../catalog/noise.csv')
