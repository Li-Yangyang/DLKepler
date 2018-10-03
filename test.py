from astropy.io import fits
import kepio, kepreduce
import glob
import re
from astropy.stats import sigma_clipped_stats
import pandas as pd
import numpy as np

cata_path = '../catalog/cumulative+noise.csv'
data_path = '/scratch/kepler_data/'
catalog_frame= pd.read_csv(cata_path, skiprows=67)

kepid = catalog_frame['kepid'].values

for i in range(len(kepid[:3])):

    filedir = kepio.pathfinder(kepid[i], data_path, '*')
    filenames = glob.glob(filedir)
    filenames.sort()
    datepat = re.compile(r'Q+\d+')
    index = 0
    for j in range(len(filenames)):
        index = datepat.findall(filenames[j])[0][1:]

    #print(catalog_frame.loc[lambda df: df.kepid == kepid[2], 'Q'+index[1]+'rms'].values, kepid[2])
    #for j in range(len(index)):
        rms = catalog_frame.loc[lambda df: df.kepid == kepid[i], 'Q'+index+'rms'].values[0]
        print("fname %s Q: %s rms %f\n" % (filenames[j], 'Q'+index, rms))
    print(kepid[i])
