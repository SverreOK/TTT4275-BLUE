import numpy as np
import matplotlib.pyplot as plt


amplitude = 1
frequency = 100000
T = 1 / frequency
angular_frequency = 2 * np.pi * frequency
phase_offset = np.pi / 8
Fs = 1000000
Ts = 1 / Fs


N = 513  # Number of samples
n0 = -(N-1)  # Starting index for samples

# Generate the signal :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :)
# Generate sample indices
n = np.arange(n0, n0 + N)

# Generate time array
t = n * Ts

# complex exponential signal
s = amplitude * np.exp(1j * (angular_frequency * t + 
phase_offset))
SNRdb = 30
SNR = 10 ** (SNRdb / 10)

# complex white gaussian noise
mean = 0
std_dev = amplitude / np.sqrt(2 * SNR)
variance = std_dev ** 2
noise = np.random.normal(mean, std_dev, N) + 1j * np.random.normal(mean, std_dev, N)

# noisy signal
x = s + noise
# x = noise

# Calculate magnitude and phase of the noisy signal
magnitude = np.abs(x)
phase = np.angle(x)
phase = np.unwrap(phase)

# Finding the blue :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :)
# fisher matrix
# J = np.zeros(N, 2)
# J[:, 0] = n * Ts
# J[:, 1] = 1
# FIM = J.T @ J / 1 # initial sigma?
# covariance = np.linalg.inv(FIM)

# Finding the blue
n = np.arange(N) + n0
H = np.column_stack((n*T, np.ones(N)))
C = np.eye(N)*1

# Compute the BLUE estimator
# BLUE = (H^T C^-1 H)^-1 H^T C^-1 phase_unwrapped
H_trans_C_inv = np.dot(H.T, np.linalg.inv(C))
blue_coefficients = np.dot(np.linalg.inv(np.dot(H_trans_C_inv, H)), np.dot(H_trans_C_inv, phase))

# The estimated frequency and phase from the BLUE estimator
estimated_omega_0T = blue_coefficients[0]
estimated_phi = blue_coefficients[1]

# The estimated signal
s_est = amplitude * np.exp(1j * (estimated_omega_0T * t + estimated_phi))

# print the original and estimated frequency and phase from the BLUE estimator
print('Original Frequency: ', angular_frequency)
print('Original Phase: ', phase_offset)
print('Estimated Frequency: ', estimated_omega_0T)
print('Estimated Phase: ', estimated_phi)


# plot original signal and noisy signal and the blue
plt.figure(figsize=(10, 6))
plt.plot(s.real, s.imag, '.', label='OG Signal', color='red')
plt.plot(x.real, x.imag, '.', label='Noisy Signal', color='blue')
plt.plot(s_est.real, s_est.imag, '.', label='Estimated Signal', color='green')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Complex Signal')
plt.legend()
plt.show()