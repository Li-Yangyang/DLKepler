import numpy as np
import pandas as pd
from astropy.io import fits
import kepio
import glob
import re

cata_path = '../catalog/cumulative.csv'
data_path = '/scratch/kepler_data/'
catalog_frame= pd.read_csv(cata_path,skiprows=65)

kepid = catalog_frame['kepid'].values

for i in range(1):
    filedir = kepio.pathfinder(kepid[i], data_path, '*')
    filenames = glob.glob(filedir)
    filenames.sort()
    datepat = re.compile(r'Q+\d+')
    index = []
    for j in range(len(filenames)):
        index.append(datepat.findall(filenames[j])[0][1:])
    d = dict(zip(index, filenames))
    newfilenames = [d[k] for k in sorted(d.keys(), key=int)]
    print(newfilenames)