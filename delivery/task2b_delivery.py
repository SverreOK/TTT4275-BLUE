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

def BLUE(SNRdb, phase_variance):
    n = np.arange(n0, n0 + N)
    t = n * Ts
    s = amplitude * np.exp(1j * (angular_frequency * t + phase_offset))

    # Noise calculation
    SNR = 10 ** (SNRdb / 10)
    std_dev = amplitude / np.sqrt(2 * SNR)
    noise = np.random.normal(0, std_dev, N) + 1j * np.random.normal(0, std_dev, N)
    x = s + noise

    phase = np.unwrap(np.angle(x))

    # Estimation using BLUE
    H = np.column_stack((n * Ts, np.ones(N)))
    C = np.eye(N) * phase_variance
    H_trans_C_inv = np.dot(H.T, np.linalg.inv(C))
    blue_coefficients = np.dot(np.linalg.inv(np.dot(H_trans_C_inv, H)), np.dot(H_trans_C_inv, phase))

    estimated_omega_0T = blue_coefficients[0]
    estimated_phi = np.mod(blue_coefficients[1], 2 * np.pi)

    s_est = amplitude * np.exp(1j * (estimated_omega_0T * t + estimated_phi))
    blue_covariance = np.linalg.inv(np.dot(H_trans_C_inv, H))
    variance_omega_0T = blue_covariance[0, 0]
    variance_phi = blue_covariance[1, 1]

    # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.plot(s.real, s.imag, '.', label='Original Signal', color='red')
    # plt.plot(x.real, x.imag, '.', label='Noisy Signal', color='blue')
    # plt.plot(s_est.real, s_est.imag, '.', label='Estimated Signal', color='green')
    # plt.xlabel('Real')
    # plt.ylabel('Imaginary')
    # plt.title(f'Complex Signal Visualization at SNR = {SNRdb} dB')
    # plt.legend()
    return estimated_omega_0T, estimated_phi, std_dev**2, phase_variance, variance_omega_0T, variance_phi

def crlb(SNRdb):
    """
    Function to calculate the Cramer-Rao Lower Bound (CRLB) for frequency and phase estimation.
    """
    SNR = 10 ** (SNRdb / 10)
    std_dev = amplitude / np.sqrt(2 * SNR)
    variance = std_dev ** 2
    P = N * (N - 1) / 2
    Q = N * (N - 1) * (2 * N - 1) / 6

    var_omega = 12 * variance / (amplitude**2 * Ts**2 * N * (N**2 - 1))
    var_phi   = 12 * variance * (n0**2 * N + 2 * n0 * P + Q) / (amplitude**2 * N**2 * (N**2 - 1))

    print(f'CRLB for Frequency at SNR={SNRdb} dB: {var_omega}')
    print(f'CRLB for Phase at SNR={SNRdb} dB: {var_phi}')

def empiricalVariance(SNRdb, phase_variance, num_simulations=100):
    omega_0T_estimates = []
    phi_estimates = []

    for _ in range(num_simulations):
        coeffs = BLUE(SNRdb, phase_variance)
        omega_0T_estimates.append(coeffs[0])
        phi_estimates.append(coeffs[1])

    # Calculate empirical variances
    var_omega_0T = np.var(omega_0T_estimates)
    var_phi = np.var(phi_estimates)

    return var_omega_0T, var_phi
    
print("{:<10} {:<25} {:<25} {:<25} {:<25}".format("SNRdb", "omega0 theoretical", "empirical", "phi theoretical", "empirical"))
for SNRdb in [-10, 0, 10, 20, 30, 40]:
    phase_variance = findPhaseVariance(SNRdb, frequency)
    blue = BLUE(SNRdb, phase_variance)
    theoretical_variance_omega_0T = blue[4]
    theoretical_variance_phi = blue[5]
    empirical_variances = empiricalVariance(SNRdb, phase_variance)
    
    print("{:<10} {:<25.2e} {:<25.2e} {:<25.2e} {:<25.2e}".format(
        f"{SNRdb} dB", theoretical_variance_omega_0T, empirical_variances[0], theoretical_variance_phi, empirical_variances[1]))