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

# Plot magnitude and phase
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, magnitude)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('Magnitude of Noisy Signal')

plt.subplot(2, 1, 2)
plt.plot(t, phase)
plt.xlabel('Time')
plt.ylabel('Phase (radians)')
plt.title('Phase of Noisy Signal')

plt.tight_layout()
plt.show()

# Plot complex signal
plt.figure(figsize=(8, 6))
plt.plot(x.real, x.imag, '.', label='Noisy Signal', color='red')


plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Complex Signal with and without Noise')
plt.grid(True)
plt.axis('equal')  # Equal aspect ratio

circle = plt.Circle((0, 0), amplitude, color='b', fill=False, linestyle='--', label='Original Signal')
plt.gca().add_artist(circle)

plt.legend()
plt.show()