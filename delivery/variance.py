import numpy as np
import matplotlib.pyplot as plt

def findVariance(SNRdb):
    amplitude = 1
    SNR = 10 ** (SNRdb / 10)
    std_dev = amplitude / np.sqrt(2 * SNR)

    variance = std_dev ** 2
    return variance

def findPhaseVariance(SNRdb, N):
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

    # find the phase_variance in phase
    expected_phase = np.angle(s)


    phase_variance = 0
    phase_adjusted = np.copy(phase)  

    for i in range(len(phase)):
        while abs(phase_adjusted[i] - expected_phase[i]) > np.pi:
            if phase_adjusted[i] - expected_phase[i] > np.pi:
                phase_adjusted[i] -= 2 * np.pi
            elif phase_adjusted[i] - expected_phase[i] < -np.pi:
                phase_adjusted[i] += 2 * np.pi
    
    # Calculate phase_variance using vectorized operations
    phase_variance = np.mean((phase_adjusted - expected_phase) ** 2)
    
    return phase_variance

def main():
    SNRdb = 0
    N = 100000
    phase_variance = findPhaseVariance(SNRdb, N)
    print('Variance: ', phase_variance)

    #plot phase_variance against SNR with log scale
    SNRdb = np.linspace(-10, 30, 100)
    phase_variance = np.zeros(100)
    for i in range(100):
        phase_variance[i] = findPhaseVariance(SNRdb[i], N)

    plt.plot(SNRdb, phase_variance)
    plt.yscale('log')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Sigma_V^2')

    plt.show()

    #plot phase variance and variance against SNR with log scale
    SNRdb = np.linspace(-10, 30, 100)
    phase_variance = np.zeros(100)
    variance = np.zeros(100)
    for i in range(100):
        phase_variance[i] = findPhaseVariance(SNRdb[i], N)
        variance[i] = findVariance(SNRdb[i])
    
    plt.plot(SNRdb, phase_variance, label='sigma_v^2')
    plt.plot(SNRdb, variance, label='sigma_w^2')
    plt.yscale('log')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Variance')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()