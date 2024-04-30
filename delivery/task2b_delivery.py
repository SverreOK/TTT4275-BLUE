import numpy as np
import matplotlib.pyplot as plt
from variance import findPhaseVariance

# Constants
amplitude = 1                                       # Amplitude
frequency = 100000                                  # Frequency
angular_frequency = 2 * np.pi * frequency           # Angular frequency
phase_offset = np.pi / 8                            # Phase offset
Fs = 1000000000                                     # Sampling frequency
Ts = 1 / Fs                                         # Sampling period
N = 513                                             # Number of samples
n0 = -256                                           # Start index
SNRdb = -10                                         # Signal to noise ratio in dB
SNR = 10 ** (SNRdb / 10)                            # Signal to noise ratio
sigma_w = amplitude / np.sqrt(2 * SNR)              # Standard deviation of noise

def generate_signal(amplitude, angular_frequency, phase_offset, N, Ts, n0):
    n = np.arange(n0, n0 + N)
    t = n * Ts
    return amplitude * np.exp(1j * (angular_frequency * t + phase_offset)), t

def add_noise(signal, SNRdb):
    SNR = 10 ** (SNRdb / 10)
    std_dev = amplitude / np.sqrt(2 * SNR)
    noise = np.random.normal(0, std_dev, len(signal)) + 1j * np.random.normal(0, std_dev, len(signal))
    return signal + noise, std_dev**2