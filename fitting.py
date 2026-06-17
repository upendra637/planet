import numpy as np
import matplotlib.pyplot as plt 
from bin import flat_data
import batman

flc = flat_data()
flux = flc.flux.value
time = flc.time.value

def mod(time):

    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = 0.00403554483570368      #time of inferior conjunction
    params.per = 3.522                    #orbital period
    params.rp = 0.14650606006179753       #planet radius (in units of stellar radii)
    params.a = 3.6383024955202954         #semi-major axis (in units of stellar radii)
    params.inc = 74.07318993651366        #orbital inclination (in degrees)
    params.ecc = 0.                       #eccentricity
    params.w = 90.                        #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"        #limb darkening model
    params.u = [0.1,0.3]                  #limb darkening coefficients [u1, u2, u3, u4]

    m = batman.TransitModel(params, time)    #initializes model

    flux = m.light_curve(params)

    return flux

mod_flux = mod(time)

plt.plot(time,flux)
plt.plot(time, mod_flux)
plt.show()