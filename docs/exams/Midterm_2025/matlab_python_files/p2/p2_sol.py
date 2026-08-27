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

maxDoppler = int(1000)

# vector of tentative Doppler frequencies to search
doppler = np.linspace(-maxDoppler, maxDoppler, int(2*maxDoppler / doppler_step) + 1)

# generate the time axis
t = np.arange(y.size) * Ts

# initialize values
IP_result = -float('inf')
doppler_estim = -float('inf')
tau_estim = -float('inf')

# loop over tentative Doppler
for fd in doppler:
    ydc = y * np.exp(-1j * 2 * np.pi * fd * t)  # y corrected for tentative Doppler
    R = abs(corr(ydc, p, 'valid', 'fft'))

    # find the maximum and its position
    tau = np.argmax(R)
    Rmax = R[tau]
    if Rmax > IP_result:
        IP_result = Rmax
        doppler_estim = fd
        tau_estim = tau

print('estimatedDelay = ', tau_estim)
print('estimatedDoppler = ', doppler_estim)

# correct the Doppler, remove the delay, and do a scatterplot
ydc = y * np.exp(-1j * 2 * np.pi * doppler_estim * t)
ydc = ydc[tau_estim:]

plt.figure()
plt.scatter(ydc.real, ydc.imag, marker='*')
plt.title('Received samples')
plt.grid()
plt.show()
