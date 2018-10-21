from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import pandas as pd
import kepler_utils
import kepler_spline
from astropy.stats import sigma_clipped_stats
from astropy.constants import R_earth, R_sun

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




#koi_ror = koi_ror[(koi_ror > 0.0) & (koi_ror < 0.1)]
#plt.hist(koi_ror, 50, facecolor = 'green')
#plt.show()

#koi_period = koi_period[(koi_period > 0.0) & (koi_period < 400)]
#plt.hist(koi_period, 50, facecolor = 'green')
#plt.show()



#koi_duration = koi_duration[(koi_duration > 0.0) & (koi_duration < 400)]
#plt.hist(koi_duration, 50, facecolor = 'green')
#plt.xlim(0, 20)
#plt.show()
data_path = '/scratch/kepler_data/kepler_Q'
file_list = glob.glob('*.fits')


#### start to read the light curve:
for i in np.arange(kepler_id.size):
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


	#create the directory to store the data
	data_path_save = '/scratch/Yinan/kepler_simulation/noise_data/'+str(kepler_id[i])+'_data/'
	#print(os.path.exists(data_path))
	if not os.path.exists(data_path_save):
		os.mkdir(data_path_save)

	for kk in np.arange(len(all_flux)):
		#plt.plot(all_time[kk], all_flux[kk]/spline[kk], marker='.', linestyle='None', color = 'blue')
		#plt.show()
		normalized_flux = all_flux[kk]/spline[kk]
		mean, median, std = sigma_clipped_stats(normalized_flux, sigma = 3.0, iters = 5)
		rs = np.random.RandomState(seed=13)
		flux1 = np.zeros(normalized_flux.size)+ median
		errors1 = std*np.ones_like(normalized_flux) # if comparable to the depth of the transit
		flux1 += errors1*rs.randn(len(normalized_flux))
		#plt.plot(all_time[kk], flux, marker='.', linestyle='None', color = 'red')
		#plt.plot(all_time[kk], spline[kk])
		#plt.show()
		#print('start to save...')
		save_file_name = data_path_save+'data_' + str(kepler_id[i]) + 'quarter_'+str(kk)+'.txt'
		print(save_file_name)
		time_slice = np.array(all_time[kk])
		#print(time_slice.shape)
		flux_slice = np.array(all_flux[kk])
		#error_slice = np.array(all_error[kk])
		spline_slice = np.array(spline[kk])
		#print(spline_slice)
		cc= np.array([time_slice, flux_slice, spline_slice]).T
		np.savetxt(save_file_name, cc)
		#print('finished')
