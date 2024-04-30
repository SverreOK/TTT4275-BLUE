from matplotlib import pyplot as plt
import numpy as np
from variance import findPhaseVariance
# Adjusting the functions based on the user's uploaded code snippets and details.

# Extracting relevant computational parts from the provided file contents
# From task2bcleaned.py - Assume it handles phase unwrapping and BLUE estimation
SNRdB_range = np.linspace(-10, 40, 50)

# Constants
amplitude = 1
frequency = 100000
# frequency = 600
angular_frequency = 2 * np.pi * frequency
phase_offset = np.pi / 8
Fs = 1000000000
Ts = 1 / Fs
N = 513  # Number of samples
n0 = -256  # Start index

def generate_signal(amplitude, angular_frequency, phase_offset, N, Ts, n0):
    """Generate complex exponential signal with time array."""
    n = np.arange(n0, n0 + N)
    t = n * Ts
    return amplitude * np.exp(1j * (angular_frequency * t + phase_offset))

def add_noise(signal, SNRdb):
    """Add complex white Gaussian noise to the signal."""
    SNR = 10 ** (SNRdb / 10)
    std_dev = amplitude / np.sqrt(2 * SNR)
    noise = np.random.normal(0, std_dev, len(signal)) + 1j * np.random.normal(0, std_dev, len(signal))
    return signal + noise, std_dev**2

def calculate_crlb(N, n0, Ts, amplitude, SNRdb):
    """Calculate and print the Cramer-Rao Lower Bound (CRLB)."""
    SNR = 10 ** (SNRdb / 10)
    std_dev = amplitude / np.sqrt(2 * SNR)
    variance = std_dev ** 2
    P = N * (N - 1) / 2
    Q = N * (N - 1) * (2 * N - 1) / 6
    var_omega = 12 * variance / (amplitude**2 * Ts**2 * N * (N**2 - 1))
    var_phi = 12 * variance * (n0**2 * N + 2 * n0 * P + Q) / (amplitude**2 * N**2 * (N**2 - 1))
    return var_omega, var_phi

def phasedifference_blue_estimator(Ts, SNRdb, N, phase_variance):
    """Perform BLUE estimation for frequency."""
    n = np.arange(n0, n0 + N)
    t = n * Ts
    s = amplitude * np.exp(1j * (angular_frequency * t + phase_offset))

    # Noise calculation
    SNR = 10 ** (SNRdb / 10)
    std_dev = amplitude / np.sqrt(2 * SNR)
    noise = np.random.normal(0, std_dev, N) + 1j * np.random.normal(0, std_dev, N)
    x = s + noise
    phases = np.angle(x)
    phase_diff = phases[1:] - phases[:-1]

    # phase_diff = np.where(phase_diff > np.pi, phase_diff - 2 * np.pi, phase_diff)
    # phase_diff = np.where(phase_diff < -np.pi, phase_diff + 2 * np.pi, phase_diff)
    phase_diff = np.array(phase_diff)
    H = Ts * np.ones((N-1, 1))
    C = np.diag(2 * phase_variance * np.ones(N-1)) - np.diag(phase_variance * np.ones(N-2), k=1) - np.diag(phase_variance * np.ones(N-2), k=-1)
    C_inv = np.linalg.inv(C)
    omega_0_hat = np.linalg.inv(H.T @ C_inv @ H) @ (H.T @ C_inv @ phase_diff)
    return np.linalg.inv(H.T @ C_inv @ H), omega_0_hat

def unwrapped_phase_blue_estimator(Ts, SNRdb, N, phase_variance):
    """
    Function to perform the Best Linear Unbiased Estimation (BLUE) on a noisy signal.
    """
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
    return variance_omega_0T, estimated_omega_0T

phase_variance = [findPhaseVariance(SNRdb, N) for SNRdb in SNRdB_range]

theoretical_diff = [
    phasedifference_blue_estimator(Ts, snr, N, phase_variance[idx])[0][0][0]
    for idx, snr in enumerate(SNRdB_range)
]

theoretical_unwrapped = [
    unwrapped_phase_blue_estimator(Ts, snr, N, phase_variance[idx])[0]
    for idx, snr in enumerate(SNRdB_range)
]

crlb_values = [
    calculate_crlb(N, n0, Ts, amplitude, snr)[0]
    for idx, snr in enumerate(SNRdB_range)
]


# Plotting the results
plt.figure(figsize=(6, 5))
plt.plot(SNRdB_range, theoretical_diff, label='Phase Difference Estimator')
plt.plot(SNRdB_range, theoretical_unwrapped, label='Unwrapped Phase Estimator')
plt.plot(SNRdB_range, crlb_values, label='CRLB')
plt.xlabel('SNR (dB)')
plt.ylabel('Variance')
plt.title('Theoretical estimator variance and CRLB')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()

# find the emirical variance for the unwrapped phase estimator
def empirical_variance_unwrapped(SNRdb, phase_variance, iterations=200):
    """Calculate empirical variance over multiple iterations."""
    estimates = []
    for _ in range(iterations):
        estimates.append(unwrapped_phase_blue_estimator(Ts, SNRdb, N, phase_variance)[1])
    print("1, ", SNRdb ,_)

    return np.var(estimates), np.mean(estimates)
# find the emirical variance for the phase difference estimator
def empirical_variance_diff(SNRdb, phase_variance, iterations=200):
    """Calculate empirical variance over multiple iterations."""
    estimates = []
    for _ in range(iterations):
        estimates.append(phasedifference_blue_estimator(Ts, SNRdb, N, phase_variance)[1])
    print("2, ", SNRdb)
    
    return np.var(estimates), np.mean(estimates)
        


empirical_variance_unwrapped_values = [empirical_variance_unwrapped(SNRdb, phase_variance[idx]) for idx, SNRdb in enumerate(SNRdB_range)]
empirical_variance_diff_values = [empirical_variance_diff(SNRdb, phase_variance[idx]) for idx, SNRdb in enumerate(SNRdB_range)]

print("hello")
# save the results as a csv file
import csv

with open('empirical_variance_unwrapped.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['SNRdb', 'Empirical Variance', 'Mean of Estimates'])
    for idx, SNRdb in enumerate(SNRdB_range):
        writer.writerow([SNRdb, empirical_variance_unwrapped_values[idx][0], empirical_variance_unwrapped_values[idx][1]])

with open('empirical_variance_diff.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['SNRdb', 'Empirical Variance', 'Mean of Estimates'])
    for idx, SNRdb in enumerate(SNRdB_range):
        writer.writerow([SNRdb, empirical_variance_diff_values[idx][0], empirical_variance_diff_values[idx][1]])

# plot the mean of the estimates against SNR
plt.figure(figsize=(6,5))
plt.plot(SNRdB_range, [mean[1] for mean in empirical_variance_unwrapped_values], label='Unwrapped Phase Estimator')
plt.plot(SNRdB_range, [mean[1] for mean in empirical_variance_diff_values], label='Phase Difference Estimator')
plt.xlabel('SNR (dB)')

# plot the empirical variances against SNR
plt.figure(figsize=(6,5))
plt.plot(SNRdB_range, [var[0] for var in empirical_variance_unwrapped_values], label='Unwrapped Phase Estimator')
plt.plot(SNRdB_range, [var[0] for var in empirical_variance_diff_values], label='Phase Difference Estimator')
plt.xlabel('SNR (dB)')
plt.ylabel('Empirical Variance')
plt.title('Empirical Variance for Unwrapped Phase and Phase Difference Estimators')
plt.yscale('log')

plt.legend()
plt.grid()
plt.show()

# plot the empirical variances and CRLB values against SNR
plt.figure(figsize=(6,5))
plt.plot(SNRdB_range, [var[0] for var in empirical_variance_unwrapped_values], label='Unwrapped Phase Estimator')
plt.plot(SNRdB_range, [var[0] for var in empirical_variance_diff_values], label='Phase Difference Estimator')
plt.plot(SNRdB_range, crlb_values, label='CRLB')
plt.xlabel('SNR (dB)')
plt.ylabel('Variance')
plt.title('Empirical Variance and CRLB Values')

plt.yscale('log')
plt.legend()
plt.grid()
plt.show()

# plot the empirical variance and theoretical variance against SNR
plt.figure(figsize=(6,5))
plt.plot(SNRdB_range, [var[0] for var in empirical_variance_unwrapped_values], label='Empirical Variance for Unwrapped Phase Estimator')
plt.plot(SNRdB_range, [var[0] for var in empirical_variance_diff_values], label='Empirical Variance for Phase Difference Estimator')
plt.plot(SNRdB_range, theoretical_diff, label='Theoretical Variance for Phase Difference Estimator')
plt.plot(SNRdB_range, theoretical_unwrapped, label='Theoretical Variance for Unwrapped Phase Estimator')
plt.xlabel('SNR (dB)')
plt.ylabel('Variance')
plt.title('Empirical and Theoretical Variance Values')
plt.yscale('log')
plt.legend()
plt.grid()

plt.show()

