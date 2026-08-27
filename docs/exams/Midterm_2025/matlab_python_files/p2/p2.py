import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import correlate as corr

# Problem 2

# Load the samples
rx = sio.loadmat('rx_preambleSequence.mat')
y = rx['rx_preambleSequence'].flatten()

pseq = sio.loadmat('preambleSequence.mat')
p = pseq['preambleSequence'].flatten()

# Define parameters
Ts = 1/4092000  # sampling time [seconds]
doppler_step = int(10)  # [Hz]
