import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import correlate as corr

# Problem 2

# Load the various vectors
prs = sio.loadmat('preamble.mat')
preamble = prs['preamble'].flatten()

pls = sio.loadmat('pulse.mat')
pulse = pls['pulse'].flatten()

rxs = sio.loadmat('rxSamples.mat')
rxSamples = rxs['rxSamples'].flatten()

txs = sio.loadmat('txSymbols.mat')
txSymbols = txs['txSymbols'].flatten()

# Define parameters
L = pulse.size
