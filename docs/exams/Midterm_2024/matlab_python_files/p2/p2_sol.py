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

# Q1
# create the waveform corresponding to the preamble
preambleWaveform = np.kron(preamble, pulse)

# search for the preamble
R = corr(rxSamples, preambleWaveform, 'valid')

# plot the corr
plt.figure()
plt.plot(abs(R), marker='o')
plt.grid()
plt.title('Correlation at Rx with the preamble')
plt.show()

ind = np.argmax(abs(R))
print('ind = ', ind)
Rmax = R[ind]
phi0 = np.angle(Rmax)
print('phi0 = ', phi0)

# there could be more bits before the preamble
startFirstBit = ind % L
print('startFirstBit = ', startFirstBit)

# Q2
rxSamplesCut = rxSamples[startFirstBit:]
excessEnd = rxSamplesCut.size % L
rxSamplesCut = rxSamplesCut[:-excessEnd]

# Q3
rxSamplesCut = rxSamplesCut*np.exp(-1j*phi0)  # phase correction

# Q4
# here we do the inner products (MF) in one shot
rxMatrix = rxSamplesCut.reshape((L, -1), order='F')
suffstat = np.dot(np.transpose(rxMatrix), np.transpose(np.conj(pulse)))

# plot the resulting constellation
plt.scatter(suffstat.real, suffstat.imag, marker='*')
plt.xlabel('Real')
plt.ylabel('Imag')
plt.title('Output of the MF')
plt.grid()
tmp = np.max(np.abs(suffstat))
plt.ylim([-tmp, tmp])
plt.show()

# Q5
# decode the bits
decSymbols = 2*(np.real(suffstat) > 0) - 1

# compute the SER
ser = np.sum(decSymbols != txSymbols)/decSymbols.size
print('ser = ', ser)
