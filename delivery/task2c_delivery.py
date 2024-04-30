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

def calculate_crlb(N, n0, Ts, amplitude, variance):
    P = N * (N - 1) / 2
    Q = N * (N - 1) * (2 * N - 1) / 6
    var_omega = 12 * variance / (amplitude**2 * Ts**2 * N * (N**2 - 1))
    var_phi = 12 * variance * (n0**2 * N + 2 * n0 * P + Q) / (amplitude**2 * N**2 * (N**2 - 1))
    return var_omega, var_phi

def blue_estimator(noisy_signal, Ts, SNRdb, N, phase_variance):
    phases = np.angle(noisy_signal)
    phase_diff = phases[1:] - phases[:-1]
    # phase_diff = np.where(phase_diff > np.pi, phase_diff - 2 * np.pi, phase_diff)
    # phase_diff = np.where(phase_diff < -np.pi, phase_diff + 2 * np.pi, phase_diff)
    phase_diff = np.array(phase_diff)
    H = Ts * np.ones((N-1, 1))
    C = np.diag(2 * phase_variance * np.ones(N-1)) - np.diag(phase_variance * np.ones(N-2), k=1) - np.diag(phase_variance * np.ones(N-2), k=-1)
    C_inv = np.linalg.inv(C)
    omega_0_hat = np.linalg.inv(H.T @ C_inv @ H) @ (H.T @ C_inv @ phase_diff)
    return omega_0_hat, np.linalg.inv(H.T @ C_inv @ H)

def empirical_variance(SNRdb, phase_variance, iterations=1000):
    estimates = [blue_estimator(add_noise(generate_signal(amplitude, angular_frequency, phase_offset, N, Ts, n0)[0], SNRdb)[0], Ts, SNRdb, N, phase_variance)[0] for _ in range(iterations)]
    return np.var(estimates), np.mean(estimates)


s, t = generate_signal(amplitude, angular_frequency, phase_offset, N, Ts, n0)
x, noise_variance = add_noise(s, SNRdb)
var_omega, var_phi = calculate_crlb(N, n0, Ts, amplitude, noise_variance)
phase_variance = findPhaseVariance(SNRdb, frequency)
omega_0_hat, omega_0_hat_variance = blue_estimator(x, Ts, SNRdb, N, phase_variance)

print('SNR:', 10**(SNRdb/10))
print('Additive noise variance:', noise_variance)
print('CRLB for angular frequency:', var_omega)
print('CRLB for phase:', var_phi)
print('Original omega_0:', angular_frequency)
print('Estimated omega_0:', omega_0_hat)
print('Theoretical estimator variance:', f"{omega_0_hat_variance[0][0]:.2e}")

emp_var = empirical_variance(SNRdb, phase_variance, 1)
print('Empirical estimator variance:', f"{emp_var[0]:.2e}")
print('Mean of estimates:', emp_var[1])
print("{:<10} {:<25} {:<25}".format("SNRdb", "omega0 theoretical", "empirical"))
for SNRdb in [-10, 0, 10, 20, 30, 40]:
    s, t = generate_signal(amplitude, angular_frequency, phase_offset, N, Ts, n0)
    x, noise_variance = add_noise(s, SNRdb)
    crlb = calculate_crlb(N, n0, Ts, amplitude, noise_variance)
    print(f'CRLB for Frequency at SNR={SNRdb} dB: {crlb[0]:.2e}')