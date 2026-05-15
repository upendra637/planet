# importing libraries
import numpy as np 
import matplotlib.pyplot as plt
from lightkurve import search_targetpixelfile

# Getting the target pixel file for a star in 
tpf = search_targetpixelfile('KIC 6922244', author = 'kepler' , quarter = 4, cadence = 'long').download()

# plotting the pixel image
tpf.plot()
plt.show()

# converting the pixel file to a lightkurve 
lc = tpf.to_lightcurve(aperture_mask = tpf.pipeline_mask)

# plotting the lightkurve 
lc.plot()
plt.show()

# getting the period
p = np.linspace(1,20,1000)
pg = lc.to_periodogram(method = 'bls' , period = p , frequency_factor = 500)
print(pg.period_at_max_power)
