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




for iplanet in np.arange(koi_duration.size):
	print(str(kepler_id[iplanet]))


	#read the flux here:
	read_data_path = '/scratch/Yinan/kepler_simulation/noise_data/'+str(kepler_id[iplanet])+'_data/'
	data_file_names = os.listdir(read_data_path)
	time_final = []
	flux_final = []
	flux_old = []
	for file_id in np.arange(len(data_file_names)):
		read_files = read_data_path +'data_' + str(kepler_id[iplanet]) + 'quarter_'+str(file_id)+'.txt'
		#print(read_files)
		data_frame = np.loadtxt(read_files)
		time_slice = data_frame[:,0]
		flux_slice = data_frame[:,1]
		spline_slice = data_frame[:,2]
		normalized_flux = flux_slice/spline_slice
		mean, median, std = sigma_clipped_stats(normalized_flux, sigma = 3.0, iters = 5)

		rs = np.random.RandomState(seed=13)
		flux1 = np.zeros(normalized_flux.size)+ median
		errors1 = std*np.ones_like(normalized_flux)
		flux1 += errors1*rs.randn(len(normalized_flux))

		time_final.append(time_slice)
		flux_final.append(flux1)



	t= np.concatenate(time_final)
	noise = np.concatenate(flux_final)
	smass = koi_smass[iplanet]
	srad = koi_srad[iplanet]
	plt.plot(t, noise,marker='.', linestyle='None', color = 'red')
	plt.show()

	print('stellar mass is %s' % smass)
	print('stellar radius is %s' % srad)

	cc =planet_simulation(smass, srad, noise, t)




