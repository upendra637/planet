# importing libraries
import numpy as np
import matplotlib.pyplot as plt 
from lightkurve import search_targetpixelfile

# Getting the target pixel file for a star in 
tpf = search_targetpixelfile('KIC 6922244', author = 'kepler' , quarter = 4, cadence = 'long')

# plotting the pixel image
tpf.plot()
plt.show()

# converting the pixel file to a lightkurve 
lc = tpf.to_lightkurve(aperture_mask = tpf.pipeline_mask)

# plotting the lightkurve 
lc.plot()
plt.show()
