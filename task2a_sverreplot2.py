import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Signal parameters
amplitude = 1
frequency = 100000
angular_frequency = 2 * np.pi * frequency
phase = np.pi / 8
sample_frequency = 1000000
period = 1 / sample_frequency

# Samples
N = 513

# Time start index
P = N * (N - 1) / 2
Q = N * (N - 1) * (2 * N - 1) / 6
t0 = P / Q

# Time array
t = np.linspace(t0, t0 + (N - 1) * period, N)

# Complex exponential signal / Original signal
s = amplitude * np.exp(1j * (angular_frequency * t + phase))
original_phase = np.unwrap(np.angle(s))

# Initialize parameters
SNR_dB_values = np.arange(-10, 20, 1)  # From -10dB to 10dB
iterations = 5000
MSE_values_average = np.zeros(len(SNR_dB_values))

# Simulation loop
for _ in range(iterations):
    MSE_values = []
    for SNRdb in SNR_dB_values:
        SNR = 10 ** (SNRdb / 10)
        std_dev = amplitude / np.sqrt(2 * SNR)
        noise = np.random.normal(0, std_dev, N) + 1j * np.random.normal(0, std_dev, N)
        x = s + noise
        noisy_phase = np.unwrap(np.angle(x))
        MSE = np.mean((noisy_phase - original_phase) ** 2)
        MSE_values.append(MSE)
    MSE_values_average += np.array(MSE_values)

# Averaging the MSE
MSE_values_average /= iterations

# Interpolation for a smooth curve
f = interp1d(SNR_dB_values, MSE_values_average, kind='quadratic')
xnew = np.linspace(min(SNR_dB_values), max(SNR_dB_values), num=400, endpoint=True)

# Plot MSE vs SNR
plt.figure(figsize=(6, 5))
plt.semilogy(xnew, f(xnew), label='Interpolated MSE')  # Logarithmic scale for plotting MSE
plt.scatter(SNR_dB_values, MSE_values_average, color='red', label='Average MSE Points')
plt.xlabel('SNR (dB)')
plt.ylabel('Mean Squared Error (radians^2)')
plt.title('Average MSE of Phase Unwrapping vs. SNR (Log Scale)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Enhanced grid for clarity on log scale
plt.show()
