# importing libraries
import numpy as np 
import matplotlib.pyplot as plt
from lightkurve import search_targetpixelfile
import batman


# Getting the target pixel file for a star in 
tpf = search_targetpixelfile('KIC 6922244', author = 'kepler' , quarter = 4, cadence = 'long').download()

# converting the pixel file to a lightkurve 
lc = tpf.to_lightcurve(aperture_mask = tpf.pipeline_mask)

# Flattening the lightcurve
flattened_lc = lc.flatten(window_length = 401)

# getting the period
p = np.linspace(1,20,10000)
pg = flattened_lc.to_periodogram(method = 'bls' , period = p , frequency_factor = 500)

# folding 
fold_lc = flattened_lc.fold(period = pg.period_at_max_power , epoch_time = pg.transit_time_at_max_power)

# binning 
bin_lc = fold_lc.bin(time_bin_size = 0.01)
x = bin_lc.phase.value

# model by batman
def model(x):
    x_min = min(x)
    x_max = max(x)
    n = len(x)

    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = -0.00403554483570368      #time of inferior conjunction
    params.per = 3.522                    #orbital period
    params.rp = 0.14650606006179753       #planet radius (in units of stellar radii)
    params.a = 3.6383024955202954         #semi-major axis (in units of stellar radii)
    params.inc = 74.07318993651366        #orbital inclination (in degrees)
    params.ecc = 0.                       #eccentricity
    params.w = 90.                        #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"        #limb darkening model
    params.u = [0.1,0.3]                  #limb darkening coefficients [u1, u2, u3, u4]

    t = np.linspace(x_min,x_max,n)  #times at which to calculate light curve
    m = batman.TransitModel(params, t)    #initializes model

    flux = m.light_curve(params)

    return t,flux

time,flux = model(x)


plt.plot(time,flux)
plt.plot(bin_lc.phase.value,bin_lc.flux.value, c = 'r')
plt.show()


