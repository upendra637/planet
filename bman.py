# importing libraries
import numpy as np 
import matplotlib.pyplot as plt
from lightkurve import search_targetpixelfile
from scipy.optimize import curve_fit
import batman


# Getting the target pixel file for a star in 
tpf = search_targetpixelfile('KIC 6922244', author = 'kepler' , quarter = 4, cadence = 'long').download()

# converting the pixel file to a lightkurve 
lc = tpf.to_lightcurve(aperture_mask = tpf.pipeline_mask)

# Flattening the lightcurve
flattened_lc = lc.flatten(window_length = 401)

# getting the period
p = np.linspace(1,100,10000)
pg = flattened_lc.to_periodogram(method = 'bls' , period = p , frequency_factor = 500)

# folding 
fold_lc = flattened_lc.fold(period = pg.period_at_max_power , epoch_time = pg.transit_time_at_max_power)

# binning 
bin_lc = fold_lc.bin(time_bin_size = 0.01)
phase = bin_lc.phase.value
flux = bin_lc.flux.value


# model by batman
params = batman.TransitParams()       #object to store transit parameters
params.t0 = 0.                        #time of inferior conjunction
params.per = 3.522                    #orbital period
params.rp = 0.1                       #planet radius (in units of stellar radii)
params.a = 10.                        #semi-major axis (in units of stellar radii)
params.inc = 87.                      #orbital inclination (in degrees)
params.ecc = 0.                       #eccentricity
params.w = 90.                        #longitude of periastron (in degrees)
params.limb_dark = "quadratic"        #limb darkening model
params.u = [0.1, 0.3]                 #limb darkening coefficients [u1, u2, u3, u4]

def model(phase, rp, a, inc,t0):

    params.a = a
    params.rp = rp
    params.inc = inc
    params.t0 = t0

    m = batman.TransitModel(params,phase)
    flux = m.light_curve(params)

    return flux

p0 = [0.1, 10. , 87. , 0.]

popt , pcov = curve_fit(
    model,
    phase,
    flux,
    p0= p0
)

# taking the fitted parameter 
b_rp  = popt[0]
b_a   = popt[1]
b_inc = popt[2]
b_t0  = popt[3]

# making the model with best parameters 
best_model = model(phase,b_rp,b_a,b_inc,b_t0)

print(
    b_rp,
    b_a,
    b_inc,
    b_t0
)

# plotting
# plt.scatter(phase,flux, label = 'real light curve ', s = 3)
# plt.plot(phase,best_model , label = 'model', c = 'r')
# plt.xlabel('phase')
# plt.ylabel('flux')
# plt.legend()
# plt.show()


