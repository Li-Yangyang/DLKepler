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


data_path = '/scratch/kepler_data/kepler_Q'
file_list = glob.glob('*.fits')

training_size = 100

info_matrix = np.zeros((training_size, 7))
postive_matrix = np.zeros((training_size, 100))
negative_matrix = np.zeros((training_size, 100))


#### start to read the light curve:
for kkk in np.arange(training_size):
	i = np.random.uniform(0, koi_smass.size - 1, 1)
	i = int(round(i[0]))
	print(kkk, i)

	all_time = []
	all_flux = []
	print('The index is '+str(i))
	print('This is '+str(kepler_id[i]))


	for quarter in np.arange(18):
		if len(str(int(kepler_id[i]))) < 8:
			file_dir=data_path+str(quarter)+'/'+'kplr00'+str(int(kepler_id[i]))+'*.fits'
		else:
			file_dir=data_path+str(quarter)+'/'+'kplr0'+str(int(kepler_id[i]))+'*.fits'
		filename = glob.glob(file_dir)

		if len(filename) > 0:
			for name in filename:
				hdu = fits.open(name)
				star_lc = hdu[1].data
				bjdrefi = hdu[1].header['BJDREFI']
				bjdreff = hdu[1].header['BJDREFF']
				time = star_lc.TIME
				time = time +bjdrefi+bjdreff -2454900
				flux = star_lc.PDCSAP_FLUX
				valid_indices = np.where(np.isfinite(flux))
				time = time[valid_indices]
				flux= flux[valid_indices]

				if time.size:
					all_time.append(time)
					all_flux.append(flux)


	all_time, all_flux = kepler_utils.split(all_time, all_flux, gap_width = 0.75)

	bkspaces = np.logspace(np.log10(0.5), np.log10(20), num = 20)
	spline = kepler_spline.choose_kepler_spline(all_time, all_flux, bkspaces, penalty_coeff = 1.0, verbose=False)[0]
	if spline is None:
		raise ValueError("faied to fit spline")

	#time_final = []
	flux_final = []

	for kk in np.arange(len(all_flux)):

		normalized_flux = all_flux[kk]/spline[kk]
		mean, median, std = sigma_clipped_stats(normalized_flux, sigma = 3.0, iters = 5)
		rs = np.random.RandomState(seed=13)
		flux1 = np.zeros(normalized_flux.size)+ median
		errors1 = std*np.ones_like(normalized_flux)
		flux1 += errors1*rs.randn(len(normalized_flux))

		#time_final.append(time_slice)
		flux_final.append(flux1)



	t= np.concatenate(all_time)
	noise = np.concatenate(flux_final)
	smass = koi_smass[i]
	srad = koi_srad[i]
	plt.plot(t, noise,marker='.', linestyle='None', color = 'red')
	plt.show()

	print('stellar mass is %s' % smass)
	print('stellar radius is %s' % srad)

	data =planet_simulation(smass, srad, noise, t)

	info_matrix[kkk, 0] = kepler_id[i]
	info_matrix[kkk, 1] = smass
	info_matrix[kkk, 2] = srad
	info_matrix[kkk, 3] = data['p'] 
	info_matrix[kkk, 4] = data['dur']
	info_matrix[kkk, 5] = data['rprs']
	info_matrix[kkk, 6] = data['ars']

	postive_matrix[kkk, :] = data['lc_signal']
	negative_matrix[kkk,:] = data['lc_no']


	#save your data here:




