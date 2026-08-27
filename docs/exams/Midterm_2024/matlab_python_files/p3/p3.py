import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import correlate as corr

# Problem 3

# Load the various vectors
rtmp = sio.loadmat('r.mat')
r = rtmp['r'].flatten()

ptmp = sio.loadmat('p.mat')
p = ptmp['p'].flatten()

# Define parameters
Ts = 1e-5  # sampling time in [s]
desiredResolution = 1  # in [Hz]
