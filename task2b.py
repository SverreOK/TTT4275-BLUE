import numpy as np
import matplotlib.pyplot as plt

SNRdb = 0

amplitude = 1
frequency = 100000
T = 1 / frequency
angular_frequency = 2 * np.pi * frequency
phase_offset = np.pi / 8
Fs = 1000000
Ts = 1 / Fs


N = 513  # Number of samples
n0 = -(N-1)/2  # Starting index for samples

# Generate the signal :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :) :)
# Generate sample indices
n = np.arange(n0, n0 + N)

# Generate time array
t = n * Ts

# complex exponential signal
s = amplitude * np.exp(1j * (angular_frequency * t + phase_offset))

# complex white gaussian noise
mean = 0
SNR = 10 ** (SNRdb / 10)
std_dev = amplitude / np.sqrt(2 * SNR)
variance = std_dev ** 2
noise = np.random.normal(mean, std_dev, N) + 1j * np.random.normal(mean, std_dev, N)

# noisy signal
x = s + noise
# x = s
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
n = np.arange(N)+n0
H = np.column_stack((n*Ts, np.ones(N)))
C = np.eye(N)*variance

# Compute the BLUE estimator
# BLUE = (H^T C^-1 H)^-1 H^T C^-1 phase_unwrapped
H_trans_C_inv = np.dot(H.T, np.linalg.inv(C))
blue_coefficients = np.dot(np.linalg.inv(np.dot(H_trans_C_inv, H)), np.dot(H_trans_C_inv, phase))

# The estimated frequency and phase from the BLUE estimator
estimated_omega_0T = blue_coefficients[0]
estimated_phi = blue_coefficients[1]

#adjust estimated_phi to be in the range of 0 to 2*pi
estimated_phi = np.mod(estimated_phi, 2*np.pi)

# The estimated signal
s_est = amplitude * np.exp(1j * (estimated_omega_0T * t + estimated_phi))

# print the original and estimated frequency and phase from the BLUE estimator
print('SNR: ', SNRdb)
print('Original Frequency: ', angular_frequency)
print('Original Phase: ', phase_offset)
print('Estimated Frequency: ', estimated_omega_0T)
print('Estimated Phase: ', estimated_phi)

# Compute the BLUE covariance matrix
blue_covariance = np.linalg.inv(np.dot(H_trans_C_inv, H))

# Extract the variances
variance_omega_0T = blue_covariance[0, 0]
variance_phi = blue_covariance[1, 1]

# Print variances
print('Variance of Estimated Frequency (omega_0T): ', variance_omega_0T)
print('Variance of Estimated Phase (phi): ', variance_phi)


# plot original signal and noisy signal and the blue
plt.figure(figsize=(10, 6))
plt.plot(s.real, s.imag, '.', label='OG Signal', color='red')
plt.plot(x.real, x.imag, '.', label='Noisy Signal', color='blue')
plt.plot(s_est.real, s_est.imag, '.', label='Estimated Signal', color='green')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Complex Signal')
plt.legend()
# plt.show()

def crlb():
    P = N * (N - 1) / 2
    Q = N * (N - 1) * (2 * N - 1) / 6

    # CRLB for frequency and phase
    var_omega = 12 * variance / (amplitude**2 * T**2 * N * (N**2 - 1))
    var_phi   = 12 * variance * (n0**2*N+2*n0*P+Q) / (amplitude**2 * N**2 * (N**2 - 1))

    print('CRLB for Frequency: ', var_omega)
    print('CRLB for Phase: ', var_phi)


crlb()