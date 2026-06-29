import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk


def flat_data(
        star,    # KIC number of the star to be analyzed or the star name
        author,  # The author of the light curve data (e.g., 'kepler', 'tess', etc.)
        cadence, # The cadence of the light curve data (e.g., 'long', 'short', etc.)
        quarter, # The quarter of the light curve data to be analyzed
        wind_len # The window length for the flattening process (default is 601)

):
    lc = lk.search_lightcurve(f'{star}' , author = author , cadence = cadence , quarter = quarter).download()

    flc = lc.flatten(window_length = wind_len)
    return flc

def bin_data(
        min_period,  # Minimum period for the periodogram
        max_period,  # Maximum period for the periodogram
        N, # Number of period points to evaluate in the periodogram
        bin_size # Size of the time bins for folding and binning the light curve data
):

    flc = flat_data()

    p = np.linspace(min_period, max_period, N)
    pg = flc.to_periodogram(method = 'bls', period = p, frequency_factor = 500)

    folc = flc.fold(period = pg.period_at_max_power, epoch_time = pg.transit_time_at_max_power)

    blc = folc.bin(time_bin_size = bin_size)

    return blc

def flat_plot():

    flc = flat_data()
    plt.figure(figsize=(10, 6))
    flc.plot()
    plt.title('Flattened Light Curve')
    plt.xlabel('Time (BJD - 2454833)')
    plt.ylabel('Normalized Flux')
    plt.grid()
    plt.show()

def bin_plot():
    
    blc = bin_data()
    plt.figure(figsize=(10, 6))
    blc.plot()
    plt.title('Binned Folded Light Curve')
    plt.xlabel('Phase')
    plt.ylabel('Normalized Flux')
    plt.grid()
    plt.show()



