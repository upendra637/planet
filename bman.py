# importing libraries
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import batman
from bin import flatNbin


binn = flatNbin('KIC 6922244', 'kepler', 'long', 5)

flattened_lc = binn.flat_data(401)
pg = binn.bls_method(1, 100, 10000)

bin_lc = binn.bin_data(0.01)
phase = bin_lc.phase.value
flux = bin_lc.flux.value
period = pg.period_at_max_power.value



# model by batman


def model(phase, rp, a, inc,t0):

    params = batman.TransitParams()       #object to store transit parameters

    params.a = a                          #semi-major axis (in units of stellar radii)
    params.rp = rp                        #planet radius (in units of stellar radii)
    params.inc = inc                      #orbital inclination (in degrees)
    params.t0 = t0                        #time of inferior conjunction
    params.per = period                   #orbital period
    params.ecc = 0.                       #eccentricity
    params.w = 90.                        #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"        #limb darkening model
    params.u = [0.1, 0.3]                 #limb darkening coefficients [u1, u2, u3, u4]

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
plt.scatter(phase,flux, label = 'real light curve ', s = 3)
plt.plot(phase,best_model , label = 'model', c = 'r')
plt.xlabel('phase')
plt.ylabel('flux')
plt.legend()
plt.show()


