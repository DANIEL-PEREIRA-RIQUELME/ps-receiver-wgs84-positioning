import numpy as np
import scipy.io as sio
from scipy.signal import correlate as corr
import matplotlib.pyplot as plt

# Problem 2

# Load the received signal and the preamble
rx = sio.loadmat('rx_signal.mat')
rx_signal = rx['rx_signal'].flatten()
               
pr = sio.loadmat('preamble.mat')
preamble = pr['preamble'].flatten()

# Define parameters
SPS = 5  # samples per symbol
Ts = 1e-5  # sampling time [seconds]

# In the following, you will search for the Doppler frequency in the range
# [-500, 500] Hz with a Doppler step to be chosen as detailed below. While
# searching for the Doppler, you should also find the sample index
# corresponding to the start of the preamble.

# Compute the Doppler step such that, after correcting with the tentative
# Doppler frequency which is the closest to the true one, the residual
# phase over the duration of the preamble is less than pi/10 radians.
# Remember that in the following you should work with the preamble at the
# sample level (as opposed to symbol level).

# create the preamble samples
preamble_samples = np.kron(preamble, np.ones(SPS))

# 2*pi*doppler_step*length(preamble_samples)*Ts = pi/10
doppler_step = 1/(2*10*preamble_samples.size*Ts)
print('doppler_step = ', doppler_step)

# Search and find the Doppler and the beginning of the preamble.
# Check point: you should find a Doppler around 380 Hz, and the beginning
# of the preamble at sample 96.

dopplerRange = np.arange(-500, 500, doppler_step)
t = np.arange(rx_signal.size) * Ts

tau = -float('inf')
IP = -float('inf')
doppler_est = -float('inf')

for fd in dopplerRange:
    rx_signal_corrected = rx_signal * np.exp(-1j * 2 * np.pi * fd * t)
    R = abs(corr(rx_signal_corrected, preamble_samples, 'valid', 'fft'))

    index = np.argmax(R)
    Rmax = R[index]

    if Rmax > IP:
        tau = index
        IP = Rmax
        doppler_est = fd
        Rs = R  # for plotting

print('foundTau = ', tau)
print('foundDoppler = ', doppler_est)

plt.figure()
plt.stem(np.arange(Rs.size), abs(Rs), '*')
plt.grid(True)
plt.title('Correlation result for the found doppler and tau')
plt.show()

# Correct the whole received signal for the Doppler
rx_signal_corrected = rx_signal * np.exp(-1j * 2 * np.pi * doppler_est * t)

# Plot (scatterplot) the received preamble (extracted from the corrected
# signal above)
preamble_corrected = rx_signal_corrected[tau:tau+preamble_samples.size]
plt.figure()
plt.scatter(preamble_corrected.real, preamble_corrected.imag, marker='*')
plt.ylabel('Imag')
plt.xlabel('Real')
plt.title('The received preamble (corrected for Doppler)')
plt.grid()
plt.show()

# At this point, the received samples are affected only by a rotation with
# a fixed phase (and AWGN).
# Determine this phase. To do so, you should compare the received preamble
# (corrected for Doppler) with the transmitted one. If you do a
# component-wise comparison, your estimation can be made more robust.
# Check point: you should find a phase around 2 radians.

phase_vector = np.angle(preamble_corrected/preamble_samples)
plt.figure()
plt.plot(phase_vector, '-*')
plt.grid(True)
plt.title('The Estimated Rotation (over the whole preamble)')
plt.show()

phase_est = np.mean(phase_vector)
print('phase_est = ', phase_est)

# Correct the received preamble with the phase obtained above, downsample
# and plot the result (scatterplot).
# If everything you did above was correct, at this point you should see a
# noisy complex-valued BPSK constellation.
preamble_corrected_rotated = preamble_corrected*np.exp(-1j * phase_est)
preamble_downsampled = preamble_corrected_rotated[0:preamble_corrected_rotated.size:SPS]

plt.figure()
plt.scatter(preamble_downsampled.real, preamble_downsampled.imag, marker='*')
plt.ylabel('Imag')
plt.xlabel('Real')
plt.title('Preamble corrected for Doppler and rotation, downsampled')
plt.grid()
plt.show()
