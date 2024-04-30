import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.misc import derivative

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
SNR_dB_values = np.arange(-20, 40, 1)  # Expanded range
iterations = 1000
MaxDiff_values_average = np.zeros(len(SNR_dB_values))

# Simulation loop
for _ in range(iterations):
    MaxDiff_values = []
    for SNRdb in SNR_dB_values:
        SNR = 10 ** (SNRdb / 10)
        std_dev = amplitude / np.sqrt(2 * SNR)
        noise = np.random.normal(0, std_dev, N) + 1j * np.random.normal(0, std_dev, N)
        x = s + noise
        noisy_phase = np.unwrap(np.angle(x))
        differences = np.abs(np.diff(noisy_phase))  # Calculate absolute differences between successive samples
        max_diff = np.max(differences)
        MaxDiff_values.append(max_diff)
    MaxDiff_values_average += np.array(MaxDiff_values)

# Averaging the maximum differences
MaxDiff_values_average /= iterations

# Interpolation for a smooth curve
f = interp1d(SNR_dB_values, MaxDiff_values_average, kind='quadratic')
xnew = np.linspace(min(SNR_dB_values), max(SNR_dB_values), num=400, endpoint=True)
ynew = f(xnew)

# Compute derivatives
dy_dx = np.gradient(ynew, xnew)  # First derivative
d2y_dx2 = np.gradient(dy_dx, xnew)  # Second derivative

# Plot Max Difference and its derivatives vs SNR
plt.figure(figsize=(12, 9))

# Original Max Difference curve
plt.subplot(311)
plt.plot(xnew, ynew, label='Interpolated Max Diff')
plt.scatter(SNR_dB_values, MaxDiff_values_average, color='red', label='Average Max Diff Points')
plt.ylabel('Max Diff (radians)')
plt.title('Max Difference and Derivatives vs. SNR')
plt.legend()
plt.grid(True)

# First Derivative
plt.subplot(312)
plt.plot(xnew, dy_dx, label='First Derivative', color='green')
plt.ylabel('First Derivative (radians/SNR dB)')
plt.legend()
plt.grid(True)

# Second Derivative
plt.subplot(313)
plt.plot(xnew, d2y_dx2, label='Second Derivative', color='orange')
plt.xlabel('SNR (dB)')
plt.ylabel('Second Derivative (radians/(SNR dB)^2)')
plt.legend()
plt.grid(True)

plt.show()