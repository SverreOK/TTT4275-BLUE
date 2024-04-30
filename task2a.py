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

t0 = P / Q

# Corrected calculation of time array
t = np.linspace(t0, t0 + (N - 1) * period, N)

# complex exponential signal
s = amplitude * np.exp(1j * (angular_frequency * t + phase))

SNRdb = 15
SNR = 10 ** (SNRdb / 10)


# complex white gaussian noise
mean = 0
std_dev = amplitude / np.sqrt(2 * SNR)

noise = np.random.normal(mean, std_dev, N) + 1j * np.random.normal(mean, std_dev, N)


# noisy signal
x = s + noise

#noisy signal projected onto the unit circle
y = x / np.abs(x)

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
plt.figure(figsize=(6, 5))
plt.plot(x.real, x.imag, '.', label='Noisy Signal', color='red', markersize=7)
plt.plot(y.real, y.imag, '.', label='Projection onto Unit Circle', color='green', markersize=10)
plt.plot(s.real, s.imag, '.', label='Original Signal', color='blue', markersize=10, marker='x')

plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Projection of Noisy Signal onto Unit Circle')
plt.grid(True)
plt.axis('equal')  # Equal aspect ratio

circle = plt.Circle((0, 0), amplitude, color='b', fill=False, linestyle='--')
plt.gca().add_artist(circle)

plt.legend()
plt.show()

#plot complex plane with error vectors from the original signal to the noisy signal
plt.figure(figsize=(8, 6))
plt.plot(x.real, x.imag, '.', label='Noisy Signal', color='red')
plt.plot(s.real, s.imag, '.', label='Original Signal', color='blue')

for i in range(N):
    plt.arrow(s.real[i], s.imag[i], x.real[i] - s.real[i], x.imag[i] - s.imag[i], head_width=0.05, head_length=0.05, fc='k', ec='k')

plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Complex Signal with Error Vectors')
plt.grid(True)
plt.axis('equal')  # Equal aspect ratio
plt.legend()
plt.show()