import numpy as np
import matplotlib.pyplot as plt
from variance import findVariance

SNRdb = 10

amplitude = 1
frequency = 100000

angular_frequency = 2 * np.pi * frequency
phase_offset = np.pi / 8
Fs = 1000000
Ts = 1 / Fs


N = 513  # Number of samples
n0 = -256  # Start index

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
print('SNR: ', SNR)
std_dev = amplitude / np.sqrt(2 * SNR)

variance = std_dev ** 2
print('Additive noise variance: ', variance*2)
print('Additive noise one sided variance: ', variance)
noise = np.random.normal(mean, std_dev, N) + 1j * np.random.normal(mean, std_dev, N)

# noisy signal
x = s + noise

# Step 1: Calculate the differences between consecutive phase estimates
phase_diff = np.angle(x[1:]) - np.angle(x[:-1])

# Step 2: Construct the design matrix H
H = Ts * np.ones((N-1, 1))

# Step 3: Construct the covariance matrix C
phase_variance = findVariance(SNRdb, 100000) 
print('Phase variance:', phase_variance)

C = np.diag(2 * phase_variance * np.ones(N-1)) - np.diag(phase_variance * np.ones(N-2), k=1) - np.diag(phase_variance * np.ones(N-2), k=-1)

# Step 4: Compute the BLUE estimate of ω0
# Invert the C matrix
C_inv = np.linalg.inv(C)
# Compute the BLUE estimate
omega_0_hat = np.linalg.inv(H.T @ C_inv @ H) @ (H.T @ C_inv @ phase_diff)

# Estimate of the frequency
frequency_hat = omega_0_hat / (2 * np.pi)

# Display the estimate
print('Original frequency:', angular_frequency)
print('Estimated frequency:', frequency_hat)

# Since the phase estimate directly from BLUE is not available, we need to use the direct ML estimate for phase as described
# Step 5 is not needed in this specific case as we do not need to find variance for BLUE