import numpy as np
import matplotlib.pyplot as plt

amplitude = 1
frequency = 100000
angular_frequency = 2 * np.pi * frequency
phase = np.pi / 8
sample_frequency = 1000000
period = 1 / sample_frequency

# amount of samples
N = 10000000

# Corrected calculation of time array
t = np.linspace(0, N * period, N)

# complex exponential signal
s = amplitude * np.exp(1j * (angular_frequency * t + phase))

SNRdb = 0
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

# find the variance in phase
expected_phase = np.angle(s)
variance = np.mean((phase - expected_phase) ** 2)
print(f'Variance: {variance}')

# plot first 100 samples of expected and actual phase
plt.figure(figsize=(10, 6))
plt.plot(t[:100], expected_phase[:100], label='Expected Phase')
plt.plot(t[:100], phase[:100], label='Actual Phase')
plt.xlabel('Time')
plt.ylabel('Phase (radians)')
plt.title('Phase of Noisy Signal')
plt.legend()
plt.tight_layout()
plt.show()

