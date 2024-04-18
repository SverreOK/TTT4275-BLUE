import numpy as np
import matplotlib.pyplot as plt


amplitude = 1
frequency = 100000
angular_frequency = 2 * np.pi * frequency
phase = np.pi / 8
sample_frequency = 1000000
period = 1 / sample_frequency

# amount of samples
N = 100

P = N * (N - 1) / 2
Q = N * (N - 1) * (2 * N - 1) / 6

n_0 = P / Q

# Corrected calculation of time array
t = np.linspace(n_0, n_0 + (N - 1) * period, N)

# complex exponential signal
s = amplitude * np.exp(1j * (angular_frequency * t + phase))

SNRdb = 10
SNR = 10 ** (SNRdb / 10)


# complex white gaussian noise
mean = 0
std_dev = amplitude / np.sqrt(2 * SNR)

noise = np.random.normal(mean, std_dev, N) + 1j * np.random.normal(mean, std_dev, N)


# noisy signal
x = s + noise

# Calculate magnitude and phase of the noisy signal
magnitude = np.abs(x)
phase = np.angle(x)

phase = np.unwrap(phase)