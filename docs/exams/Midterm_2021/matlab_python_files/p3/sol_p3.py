import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Problem 3

# Load the received signal and the C/A code
rx = sio.loadmat('rx_signal_gps.mat')
rx_signal_gps = rx['rx_signal_gps'].flatten()

ca = sio.loadmat('ca_code.mat')
ca_code = ca['ca_code'].flatten()

# In the following we ask you to implement a matched filter receiver. The
# matched filter is matched to the pulse shape used by the satellite.
# Please follow strictly the instructions given below.

# Specify the matched filter
h_mf = np.flip(ca_code)

# Pass the received signal through the matched filter
y = np.convolve(rx_signal_gps, h_mf)

# Plot the eye diagram at the output of the matched filter
y = y[0: y.size - (y.size % (2*ca_code.size))]
ye = np.reshape(y, (2*ca_code.size, int(y.size/(2*ca_code.size))), 'F')
xe = np.arange(2*ca_code.size)

plt.figure()
plt.plot(xe, ye)
plt.title('The eye diagram at the output of the matched filter')
plt.grid()
plt.show()

# Write a code that estimates the delay of the channel, based on the eye
# diagram. For simplicity, you can assume that the eye diagram is
# symmetrical with respect to the horizontal axis.
min_over_rows = abs(ye).min(axis=1)
indexMax = np.argmax(min_over_rows)
print('indexMax = ', indexMax)

# Once you find the estimate of the delay, use it to downsample the matched
# filter output to generate the sufficient statistics
y_down = y[indexMax:y.size:ca_code.size]

plt.figure()
plt.plot(y_down, '*')
plt.title('The sufficient statistics')
plt.grid()
plt.show()

# Use the sufficient statistics to estimate the bits transmitted by the
# satellite.
estimated_bits = 2*(y_down > 0) - 1

# If everything you did above was correct, you should obtain the bits
# stored in transmitted_bits.mat
tb = sio.loadmat('transmitted_bits.mat')
transmitted_bits = tb['transmitted_bits'].flatten()

number_of_errors = np.sum(estimated_bits != transmitted_bits)
print('number_of_errors = ', number_of_errors)
