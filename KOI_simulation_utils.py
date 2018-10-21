import batman 
import astropy.units as u
from astropy.constants import G, R_sun, M_sun, R_jup, M_jup, R_earth, M_earth
import matplotlib.pyplot as plt
import numpy as np
import math
import time
#import kepler_utils
#import kepler_spline


def batman_planet(t0, per, rp, a, inc, ecc, w, t):
	params = batman.TransitParams() # object to store the transit parameters

	params.t0 = t0 # time of inferior conjunction 
	params.per = per # orbital period (days)
	params.rp = rp # planet radius (in units of stellar radii)

	params.a = a # semi-major axis (in units of stellar radii)
	params.inc = inc  # orbital inclination (in degrees)
	params.ecc = ecc # eccentricity
	params.w = w # longitude of periastron (in degrees), 90 for circular
	params.u = [0.1, 0.3] # limb darkening coefficients
	params.limb_dark = "quadratic" # limb darkening model


	m = batman.TransitModel(params, t) # initializes the model
	f = m.light_curve(params)
	return f
