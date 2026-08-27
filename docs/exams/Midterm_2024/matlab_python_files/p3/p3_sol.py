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

# see what we have in r
plt.scatter(r.real, r.imag, marker='*')
plt.xlabel('Real')
plt.ylabel('Imag')
plt.title('Received signal')
plt.grid()
plt.show()

# size of the FFT
N = np.ceil(1/(Ts*desiredResolution))

pFft = np.fft.fft(p, int(N))
rFft = np.fft.fft(r, int(N))

R = corr(np.abs(rFft), np.abs(pFft), 'full')

# display to see how good is the correlation
plt.figure()
plt.plot(R)
plt.grid()
plt.title('Correlation Result')
plt.show()

indexMax = np.argmax(R)
trueIndex = indexMax - rFft.size + 1  # index 0 corresponds to d = 0
print('trueIndex = ', trueIndex)

# check the resolution
frequencyResolution = 1/(N*Ts)
print('frequencyResolution = ', frequencyResolution)

estimatedDoppler = trueIndex*frequencyResolution
print('estimatedDoppler = ', estimatedDoppler)

# correct the signal y and plot it
timeLine = np.arange(r.size)*Ts
rCorrected = r*np.exp(-1j*2*np.pi*estimatedDoppler*timeLine)

plt.scatter(rCorrected.real, rCorrected.imag, marker='*')
plt.xlabel('Real')
plt.ylabel('Imag')
plt.title('Received signal corrected for Doppler')
plt.grid()
tmp = np.max(np.abs(rCorrected))
plt.ylim([-tmp, tmp])
plt.show()
