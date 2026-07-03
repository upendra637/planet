import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import LombScargle

class flatNbin:
    
    def __init__(self, star, author, cadence, quarter):
        self.star = star        # KIC number of the star to be analyzed or the star name
        self.author = author    # The author of the light curve data (e.g., 'kepler', 'tess', etc.)
        self.cadence = cadence  # The cadence of the light curve data (e.g., 'long', 'short', etc.)
        self.quarter = quarter  # The quarter of the light curve data to be analyzed

        self.lc = lk.search_lightcurve(self.star , author = self.author , cadence = self.cadence , quarter = self.quarter).download()




    def flat_data(self, wind_len):

        self.wind_len = wind_len


        flc = self.lc.flatten(window_length = self.wind_len)
        return flc
    

    def bls_method(self, min_period, max_period, N):
        
        self.min_period = min_period    # Minimum period for the periodogram
        self.max_period = max_period    # Maximum period for the periodogram
        self.N = N                      # Number of period points to evaluate in the periodogram

        flc = self.flat_data(self.wind_len)

        p = np.linspace(self.min_period, self.max_period, self.N)
        pg = flc.to_periodogram(method = 'bls', period = p, frequency_factor = 500)

        return pg
    
    
    def lambscargle_method(self, min_period):
        
        ls_estimate = LombScargle(self.lc.time, self.lc.flux) # this creates an instance of the LombScargle class from the astropy.timeseries module, which is used to perform a Lomb-Scargle periodogram analysis on the light curve data. The Lomb-Scargle periodogram is a method for detecting periodic signals in unevenly spaced time series data, such as astronomical light curves. The self.lc.time and self.lc.flux arrays are passed as inputs to the LombScargle constructor, allowing it to analyze the time and flux data of the light curve.

        min_freq = 1 / (self.lc.time.max() - self.lc.time.min())
        max_freq = 1 / min_period

        xf, yf = ls_estimate.autopower( # this computes the Lomb-Scargle periodogram for the light curve data. The autopower method automatically determines the frequencies at which to evaluate the periodogram, based on the specified minimum and maximum frequencies. The minimum frequency is set to the inverse of the total time span of the light curve (i.e., 1 divided by the difference between the maximum and minimum time values), while the maximum frequency is set to the inverse of the minimum period (i.e., 1 divided by min_period). The resulting xf array contains the frequencies at which the periodogram was evaluated, and the yf array contains the corresponding power values of the periodogram at those frequencies.
            minimum_frequency=min_freq, maximum_frequency=max_freq
        )

        peak_freq = xf[np.argmax(yf)]
        peak_per = np.max([1.0 / peak_freq, 1.001 * min_period]) # this identifies the frequency corresponding to the highest power in the periodogram (i.e., the most significant periodic signal) and calculates the corresponding period. The peak_freq variable is set to the frequency at which the periodogram power is maximized, and the peak_per variable is calculated as the maximum of two values: 1.0 divided by peak_freq (which gives the period corresponding to the identified frequency) and 1.001 times min_period (which ensures that the estimated period is slightly larger than the minimum period). This approach helps to avoid underestimating the oscillation period, especially if the identified frequency is close to the maximum frequency limit set by min_period.

        return peak_per



    def bin_data(self,bin_size ):
        
        
        self.bin_size = bin_size        # Size of the time bins for folding and binning the light curve data

        flc = self.flat_data(self.wind_len)

        pg = self.bls_method(self.min_period, self.max_period, self.N)

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



