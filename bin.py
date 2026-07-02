import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

class flatNbin:
    
    def __init__(self, star, author, cadence, quarter):
        self.star = star        # KIC number of the star to be analyzed or the star name
        self.author = author    # The author of the light curve data (e.g., 'kepler', 'tess', etc.)
        self.cadence = cadence  # The cadence of the light curve data (e.g., 'long', 'short', etc.)
        self.quarter = quarter  # The quarter of the light curve data to be analyzed



    def flat_data(self, wind_len):

        self.wind_len = wind_len

        lc = lk.search_lightcurve(self.star , author = self.author , cadence = self.cadence , quarter = self.quarter).download()

        flc = lc.flatten(window_length = self.wind_len)
        return flc

    def bin_data(self,min_period, max_period, N, bin_size ):
        
        self.min_period = min_period    # Minimum period for the periodogram
        self.max_period = max_period    # Maximum period for the periodogram
        self.N = N                      # Number of period points to evaluate in the periodogram
        self.bin_size = bin_size        # Size of the time bins for folding and binning the light curve data

        flc = self.flat_data(self.wind_len)

        p = np.linspace(self.min_period, self.max_period, self.N)
        pg = flc.to_periodogram(method = 'bls', period = p, frequency_factor = 500)

        folc = flc.fold(period = pg.period_at_max_power, epoch_time = pg.transit_time_at_max_power)

        blc = folc.bin(time_bin_size = self.bin_size)

        return blc

    def flat_plot(self):

        flc = self.flat_data(self.wind_len)
        plt.figure(figsize=(10, 6))
        flc.plot()
        plt.title('Flattened Light Curve')
        plt.xlabel('Time (BJD - 2454833)')
        plt.ylabel('Normalized Flux')
        plt.grid()
        plt.show()

    def bin_plot(self):
        
        blc = self.bin_data(self.min_period, self.max_period, self.N, self.bin_size)
        plt.figure(figsize=(10, 6))
        blc.plot()
        plt.title('Binned Folded Light Curve')
        plt.xlabel('Phase')
        plt.ylabel('Normalized Flux')
        plt.grid()
        plt.show()



