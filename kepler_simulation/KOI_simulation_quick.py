import batman
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import glob
import pandas as pd
import kepler_utils
from KOI_simulation_utils import *
from astropy.stats import sigma_clipped_stats
from astropy.constants import R_earth, R_sun
from kepler_simu_utils import *
import h5py
import matplotlib.pylab as plt
catalog_name = '/scratch/yyl/catalog/cumulative.csv'
catalog_frame= pd.read_csv(catalog_name,skiprows=65)



kepler_id = catalog_frame['kepid'].values
koi_period = catalog_frame['koi_period'].values

koi_duration = catalog_frame['koi_duration'].values
koi_depth = catalog_frame['koi_depth'].values
koi_ror = catalog_frame['koi_ror'].values
koi_sma = catalog_frame['koi_sma'].values
koi_incl = catalog_frame['koi_incl'].values
koi_srad = catalog_frame['koi_srad'].values
koi_smass = catalog_frame['koi_smass'].values


data_path = '/scratch/kepler_data/kepler_Q'
file_list = glob.glob('*.fits')

training_size = 10

info_matrix = np.zeros((training_size, 8))
postive_matrix = np.zeros((training_size, 200))
negative_matrix = np.zeros((training_size, 200))


#### start to read the light curve:
for kkk in np.arange(training_size):
	i = np.random.uniform(0, koi_smass.size - 1, 1)
	i = int(round(i[0]))
	print(kkk, i)

	smass = koi_smass[i]
	srad = koi_srad[i]

	if not np.isnan(smass):

		print('stellar mass is %s' % smass)
		print('stellar radius is %s' % srad)

		data = planet_simulation_simple(smass, srad)

		info_matrix[kkk, 0] = kepler_id[i]
		info_matrix[kkk, 1] = smass
		info_matrix[kkk, 2] = srad
		info_matrix[kkk, 3] = data['p'] 
		info_matrix[kkk, 4] = data['dur']
		info_matrix[kkk, 5] = data['rprs']
		info_matrix[kkk, 6] = data['ars']
		info_matrix[kkk, 7] = data['snr']

		postive_matrix[kkk, :] = data['lc_signal']
		negative_matrix[kkk,:] = data['lc_none']
		#print(np.shape(postive_matrix[kkk, :]))
		t = np.linspace(0,200,200)
		#print(t, postive_matrix[kkk, :])
		plt.scatter(t, postive_matrix[kkk, :],marker='.')
		print(data['rms'])
		plt.scatter(t, 1-np.ones(200)*3*data['rms'], color='red', marker='.')
		plt.show()

#save your data here:
with h5py.File('data_set_for_exoplanet.hdf5','w') as hf:
	hf.create_dataset("pos", data = postive_matrix)
	hf.create_dataset("neg", data = negative_matrix)
	hf.create_dataset("info", data = info_matrix)


