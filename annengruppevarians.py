import numpy as np

N = 10000
frequency = 100000
amplitude = 1
angular_frequency = 2 * np.pi * frequency
phase = np.pi / 8
sample_frequency = 1000000
period = 1 / sample_frequency


SNRdb = 0
SNR = 10 ** (SNRdb / 10)

sigma = 1 / np.sqrt(2 * SNR)

print((sigma**2)*2)

def x(n):
    w_real = np.random.normal(0, sigma, N)
    w_imag = np.random.normal(0, sigma, N)
    w = w_real + 1j * w_imag
    return np.exp(1j * (angular_frequency * n * period + phase)) + w

def x_linear(n):
    v = np.random.normal(0, np.sqrt(0.7672118198811011), N)
    return np.exp(1j * (angular_frequency * n * period + phase+v))

original_signal = x(np.arange(N))
original_signal_variance = np.var(original_signal)
print('Original signal variance: ', original_signal_variance)

noisy_signal = x_linear(np.arange(N))
noisy_signal_variance = np.var(noisy_signal)

print('Noisy signal variance: ', noisy_signal_variance)