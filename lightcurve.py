# importing libraries
import numpy as np 
import matplotlib.pyplot as plt
from lightkurve import search_targetpixelfile

# Getting the target pixel file for a star in 
tpf = search_targetpixelfile('KIC 6922244', author = 'kepler' , quarter = 4, cadence = 'long').download()

# converting the pixel file to a lightkurve 
lc = tpf.to_lightcurve(aperture_mask = tpf.pipeline_mask)

# Flattening the lightcurve
flattened_lc = lc.flatten(windows_length = 401)

# getting the period
p = np.linspace(1,20,1000)
pg = flattened_lc.to_periodogram(method = 'bls' , period = p , frequency_factor = 500)

# folding 
fold_lc = flattened_lc.fold(period = pg.period_at_max_power , epoch_time = pg.transit_time_at_max_power)

# binning 
bin_lc = fold_lc(time_bin_size = 0.01)
