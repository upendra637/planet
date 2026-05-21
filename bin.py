import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

lc = lk.search_lightcurve('kepler-69' , author = 'kepler' , cadence = 'long' , quarter = 4).download()

flc = lc.flatten(window_length = 901)

p = np.linspace(1,20,10000)
pg = flc.to_periodogram(method = 'bls', period = p, frequency_factor = 500)

folc = flc.fold(period = pg.period_at_max_power, epoch_time = pg.transit_time_at_max_power)

blc = folc.bin(time_bin_size = 0.01)

folc.plot()
blc.plot()
plt.show()