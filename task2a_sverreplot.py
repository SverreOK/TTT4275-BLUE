import numpy as np
import matplotlib.pyplot as plt

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

# Initialize SNRdb and mean
SNRdb = -20
mean = 0

# Create an array for the sample numbers
samples = np.arange(N)

plt.figure(figsize=(6,5))

while SNRdb != 30:
    # Increment SNRdb by 10
    SNRdb += 10
    SNR = 10 ** (SNRdb / 10)
    std_dev = amplitude / np.sqrt(2 * SNR)
    noise = np.random.normal(mean, std_dev, N) + 1j * np.random.normal(mean, std_dev, N)
    x = s + noise
    phase = np.angle(x)
    phase = np.unwrap(phase)
    #plt.plot(samples, phase, label=f'SNRdb =  {SNRdb}')

phase = np.angle(s)
#phase = np.unwrap(phase)
plt.plot(samples, phase, label='Original signal')

plt.xlabel('Samples')
plt.ylabel('Phase (radians)')
plt.title('Wrapped Phase of Noisy Signal')
plt.legend()
plt.xlim(0, 30)
plt.grid()
plt.show()