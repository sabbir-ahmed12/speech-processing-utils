import numpy as np
from scipy.io.wavfile import write

def addnoise(signal, snr, *args, awgn=False):
    '''
    Returns a noisy signal having certain SNR after addition of signal and noise

            Parameters:
                    signal (numpy.ndarray): An audio signal
                    snr (int)             : A target SNR 
                    *args (numpy.ndarray) : A noise, if awgn is False
                    awgn (bool)           : If True, adds white gaussian noise
            Returns:
                    noisy_signal (numpy.ndarray): Addition of signal and noise
    '''

    # output_signal_snr = 20*np.log10(np.linalg.norm(signal) /
    #                          np.linalg.norm(signal-noise))

    if awgn:
        signal_power = signal ** 2
        avg_signal_power = np.mean(signal_power)
        avg_signal_powerdB = 10 * np.log10(avg_signal_power)

        # Calculate noise and converting to linear scale
        avg_noise_powerdB = avg_signal_powerdB - snr
        avg_noise_power = 10 ** (avg_noise_powerdB / 10)

        # Generate white noise
        noise = np.random.normal(0, np.sqrt(
            avg_noise_power), len(signal_power))

        noisy_signal = signal + noise

        return noisy_signal

    else:
        signal_length = len(signal)
        noise = args[0]
        noise_length = len(noise)

        if signal_length > noise_length:
            ValueError('Error: signal length is greater than noise length')

        # Generate a random start location in the masker(noise) signal
        start = np.random.randint(1, (1+noise_length-signal_length))

        # Extract random section of the masker signal
        noise = noise[start: start+signal_length]

        # Scale the masker w.r.to target at a desired SNR level
        noise = noise / np.linalg.norm(noise) * \
            np.linalg.norm(signal) / 10.0**(0.05*snr)

        # Generate the noisy signal
        noisy_signal = signal + noise

        return noisy_signal


def generate_wav_file(noisy_signal):
    write('noisy_signal.wav', 44100, noisy_signal)
