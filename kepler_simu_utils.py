import numpy as np
import glob
from astropy.io import fits
import pandas as pd
from astropy import units as u
from astropy.constants import G
from astropy import constants as const
from KOI_simulation_utils import *
import batman
import matplotlib.pyplot as plt
import os
import kepler_utils
from KOI_simulation_utils import *
from astropy.stats import sigma_clipped_stats
from astropy.constants import R_earth, R_sun


def incl_extract(period, duration, ars, rprs, method = 'approx'):

	#period (days)
	#duration (hrs)
	#semi-major axis (in unit of stellar radii)
	#planet size (in unit of stellar daii)
	#the equation of inclination angle is adopted from Seager & Mallen-Ornelas 2003 ApJ eqauation (16)

	duration = duration / 24.0

	if method == 'approx':
		incl = np.arccos(((1+rprs)**2 - (duration*np.pi*ars/period)**2)**0.5/ars)

	if method == 'analytical':
		fraction1 = (ars*np.sin(duration*np.pi/period))**2 - (1+ rprs)**2
		fraction2 = (ars*np.sin(duration*np.pi/period))**2 - ars**2
		#print(fraction1)
		#print(fraction2)
		#print('here:', (fraction1/fraction2))

		incl = np.arccos( (fraction1/fraction2) )

	incl = incl*180.0/np.pi

	return incl

'''
doing the simulation here:
We only use the stellar properties from the real observation as the initial parameters.
For a certain star, there are three parameters important.
Stellar type, which is related to the noise level.
Stellar radii and stellar mass.

there are also three indpendent parameters generated from the uniform distribution:
planet size (in unit of stellar radii)
transit duration (hrs)
period (days)

'''

def check_snr(period, t0, time_array, flux, duration):

	time_array = kepler_utils.phase_fold_time(time_array, period, t0)
	sorted_i = np.argsort(time_array)
	time_array = time_array[sorted_i]
	flux = flux[sorted_i]

	local_view = kepler_utils.local_view(time_array, flux, period, duration)

	t_min= duration*-3.0
	t_max= duration*3.0

	time_array=np.linspace(t_min, t_max, local_view.size)#*24.0
	#start to measure the SNR around the signal.
	index_snr = np.argwhere((time_array > duration*1.0) | (time_array < duration*-1.0))
	mean_snr, median_snr, std_snr = sigma_clipped_stats(local_view[index_snr])

	index_profile = np.argwhere((time_array <= duration*1.0) & (time_array >= duration*-1.0))

	if local_view[index_profile].min() <= median_snr - 3*std_snr:

		return 1.0
	else:
		return 0.0

def transit_profile(period, t0, time_array, flux, duration):

	time_array = kepler_utils.phase_fold_time(time_array, period, t0)
	sorted_i = np.argsort(time_array)
	time_array = time_array[sorted_i]
	flux = flux[sorted_i]

	local_view = kepler_utils.local_view(time_array, flux, period, duration)

	t_min= duration*-3.0
	t_max= duration*3.0

	time_array=np.linspace(t_min, t_max, local_view.size)

	data_lc = dict()
	data_lc['time'] = time_array
	data_lc['flux'] = local_view
	return data_lc


def planet_simulation(smass, srad, noise, time_array):

	#smass: stellar mass in unit of M_sun
	#srad: stellar radii in unit of R_sun
	#noise array
	#use kepler third law to calculate the semi major axis, but only valid when inclination is close to 90, need to imporve this part.

	injection = 0.0
	iteration_step = 0.0

	while injection == 0.0:
		print('this is Iteration %s' % iteration_step)

		#change here:
		period = np.random.uniform(10,100, 1)
		duration = np.random.uniform(2, 10, 1)

		###############
		rprs = np.random.uniform(0.001,0.1, 1)
		period_s = period * 24*3600 # in unit of second

		sm_axis = (G*smass*const.M_sun*(period_s*u.s)**2/(4*np.pi**2)) **(1.0/3)/const.au #in unit of au
		sm_axis =sm_axis.value


		ars = (sm_axis*const.au/(srad*const.R_sun)).value

		#print(period, duration, ars, rprs)

		incl = incl_extract(period, duration, ars, rprs, method = 'analytical')

		t0 =0.0
		#print(period, duration, ars, rprs)

		kepler_lc = batman_planet(t0, period, rprs,ars, incl, 0, 90, time_array)
		kepler_lc_noise = kepler_lc*noise
		#plt.plot(time_array, kepler_lc_noise, marker='.', linestyle='None', color = 'red')
		#plt.show()

		injection = check_snr(period, t0, time_array, kepler_lc_noise, duration)
		iteration_step = iteration_step+1


	print('this is the orbital period inserted %s (days)' % period)
	print('this is the duration inserted %s (hrs)' % duration)
	print('this is the semi-major axis %s (stellar radius)' % ars)
	print('this is the planet radius %s (stellar radius)' % rprs)
	print('this is the inclination angle %s' % incl)

	profile = transit_profile(period, t0, time_array, kepler_lc_noise, duration)
	profile1 = transit_profile(period, t0, time_array, noise, duration)

	plt.plot(profile['time'], profile['flux'], marker='.', linestyle='None', color = 'red')
	plt.plot(profile1['time'], profile1['flux'], marker='.', linestyle='None', color = 'blue')
	plt.show()
	#print(profile1['flux'].size)

	data = dict()
	data['lc_signal'] = profile['flux']
	data['p'] = period
	data['dur'] = duration
	data['rprs'] = rprs
	data['ars'] = ars
	data['lc_no'] =  profile1['flux']
	return data



def planet_simulation_simple(smass, srad):

	#smass: stellar mass in unit of M_sun
	#srad: stellar radii in unit of R_sun
	#noise array
	#use kepler third law to calculate the semi major axis, but only valid when inclination is close to 90, need to imporve this part.



	#change here:
	period = np.random.uniform(10,100, 1)
	duration = np.random.uniform(2, 10, 1)

	###############
	rprs = np.random.uniform(0.001,0.1, 1)
	period_s = period * 24*3600 # in unit of second

	sm_axis = (G*smass*const.M_sun*(period_s*u.s)**2/(4*np.pi**2)) **(1.0/3)/const.au #in unit of au
	sm_axis =sm_axis.value


	ars = (sm_axis*const.au/(srad*const.R_sun)).value

	

	incl = incl_extract(period, duration, ars, rprs, method = 'analytical')

	t0 =0.0
	
	#####this is the view window in unit of hours
	low_t = -2*24#/2.0*6
	up_t = 2*24#/2.0*6
	time_array = np.linspace(low_t,up_t, num = 200)/24.0

	kepler_lc = batman_planet(t0, period, rprs,ars, incl, 0, 90, time_array)
	depth = kepler_lc.max() - kepler_lc.min()

	snr = np.random.uniform(3.0,7.0,1)



	rs = np.random.RandomState(seed=13)
	errors = depth/snr*np.ones_like(kepler_lc)#*0.0
	noise = errors*rs.randn(len(kepler_lc))+1.0


	kepler_lc_noise = kepler_lc*noise
	#plt.plot(time_array, kepler_lc_noise, marker='.', linestyle='None', color = 'red')
	#plt.show()



	#print('this is the orbital period inserted %s (days)' % period)
	#print('this is the duration inserted %s (hrs)' % duration)
	#print('this is the semi-major axis %s (stellar radius)' % ars)
	#print('this is the planet radius %s (stellar radius)' % rprs)
	#print('this is the inclination angle %s' % incl)
	#print('this is the SNR %s ' % snr)

	noise_new = errors*rs.randn(len(kepler_lc))+1.0

	#plt.plot(time_array*24.0, kepler_lc_noise, marker='.', linestyle='None', color = 'red')
	#plt.plot(time_array*24.0, kepler_lc, color = 'black')
	#plt.plot(time_array*24.0, noise_new, marker='.', linestyle='None', color = 'blue')
	#plt.show()

	data = dict()
	data['lc_signal'] = kepler_lc_noise
	data['lc_none'] = noise_new
	data['p'] = period
	data['dur'] = duration
	data['rprs'] = rprs
	data['ars'] = ars
	data['snr'] = snr
	return data


'''
#test the parameter here:
catalog_frame1= pd.read_csv('/scratch/Yinan/mcmc_fit_for_jian/KOI_simulation/KOI_torres_name.txt',sep='\t')
catalog1 = catalog_frame1.as_matrix()
KOI_id=catalog1[:,0]
kepler_id=catalog1[:,2]


catalog_frame2= pd.read_csv('/scratch/Yinan/mcmc_fit_for_jian/KOI_simulation/KOI_torres_info.txt',sep='\t')
catalog2 = catalog_frame2.as_matrix()

tce_rp = catalog2[:,1]
tce_period=catalog2[:, 2]
tce_t0 = catalog2[:,3]
tce_rprs = catalog2[:,4] #planet radius in stellar radii


catalog_frame3= pd.read_csv('/scratch/Yinan/mcmc_fit_for_jian/KOI_simulation/KOI_torres_duration.txt',sep='\t')
catalog3 = catalog_frame3.as_matrix()

tce_major_axis = catalog3[:,1] # semi major axis in stellar radii
tce_duration = catalog3[:,3]
tce_inclination = catalog3[:,6] #inclination

 #incl_extact(period, duration, ars, rprs, method = method):

for i in np.arange(tce_duration.size):
	incl1 = incl_extract(tce_period[i], tce_duration[i], tce_major_axis[i], tce_rprs[i], method = 'approx')
	incl2 = incl_extract(tce_period[i], tce_duration[i], tce_major_axis[i], tce_rprs[i], method = 'analytical')
	print(tce_inclination[i], incl1, incl2)
'''
