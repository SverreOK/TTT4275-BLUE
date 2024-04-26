import numpy as np
import matplotlib.pyplot as plt

def findVariance(SNRdb, N):
    amplitude = 1
    frequency = 100000
    angular_frequency = 2 * np.pi * frequency
    phase = np.pi / 8
    sample_frequency = 1000000
    period = 1 / sample_frequency


    # Corrected calculation of time array
    t = np.linspace(0, N * period, N)

    # complex exponential signal
    s = amplitude * np.exp(1j * (angular_frequency * t + phase))

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

    plt.plot(phase[:200])
    plt.plot(expected_phase[:200])
    plt.show()
    variance = 0
    phase_adjusted = np.copy(phase)  

    for i in range(len(phase)):
        while abs(phase_adjusted[i] - expected_phase[i]) > np.pi:
            if phase_adjusted[i] - expected_phase[i] > np.pi:
                phase_adjusted[i] -= 2 * np.pi
            elif phase_adjusted[i] - expected_phase[i] < -np.pi:
                phase_adjusted[i] += 2 * np.pi
    
    # Calculate variance using vectorized operations
    variance = np.mean((phase_adjusted - expected_phase) ** 2)

    # variance = np.mean((phase - expected_phase) ** 2)
    
    #plot the first 200 values of the phase and expected phase together
    plt.plot(phase_adjusted[:200])
    plt.plot(expected_phase[:200])
    plt.show()

    


    return variance

def main():
    SNRdb = 0
    N = 100000
    variance = findVariance(SNRdb, N)
    print('Variance: ', variance)

if __name__ == '__main__':
    main()