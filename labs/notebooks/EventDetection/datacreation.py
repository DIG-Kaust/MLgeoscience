import math
import numpy as np
from scipy.stats import skewnorm
from scipy.signal import butter, filtfilt


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Butterworth bandpass filter

    Parameters
    ----------
    data : :obj:`np.array`
        Data to be filtered
    lowcut : :obj:`float`
        Low cut frequency
    highcut : :obj:`float`
        High cut frequency
    fs : :obj:`float`
        Sampling frequency
    order : :obj:`int`, optional
        Filter order

    Returns
    -------
    filtdata : :obj:`np.array`
        Filtered data

    """
    # create filter
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='bandpass')
    # filter data
    filtdata = filtfilt(b, a, data)
    return filtdata


def pdf_snr(snr_min, snr_max, snr_skew, snr_loc):
    """Pdf SNR

    Compute SNR skew-normal probability density function

    Parameters
    ----------
    snr_min : :obj:`np.array`
        Minimum allowed SNR
    snr_max : :obj:`float`
        Maximum allowed SNR
    snr_skew : :obj:`float`
        SNR skewness
    snr_loc : :obj:`float`
        SNR mean

    Returns
    -------
    snr : :obj:`np.array`
        SNR axis
    pdf : :obj:`np.array`
        PDF of SRN

    """
    # define distribution
    snr_distr = skewnorm(snr_skew, loc=snr_loc)
    
    # create pdf
    snr = np.linspace(0.8*snr_min, 1.2*snr_max, 100)
    pdf = snr_distr.pdf(snr)
    return snr, pdf
    

def draw_snr(snr_min, snr_max, snr_skew, snr_loc):
    """Draw SNR

    Draw SNR from skew-normal distribution

    Parameters
    ----------
    snr_min : :obj:`np.array`
        Minimum allowed SNR
    snr_max : :obj:`float`
        Maximum allowed SNR
    snr_skew : :obj:`float`
        SNR skewness
    snr_loc : :obj:`float`
        SNR mean

    Returns
    -------
    snr : :obj:`float`
        SNR

    """
    # define distribution
    snr_distr = skewnorm(snr_skew, loc=snr_loc)

    # draw sample until it is within define min and max values
    snr = -100
    while snr < snr_min or snr > snr_max:
        snr = snr_distr.rvs()
    return snr


def ricker(fc, length=0.4, dt=0.002):
    """Ricker wavelet

    Parameters
    ----------
    fc : :obj:`float`
        Central frequency
    length : :obj:`float`
        Time lenght
    dt : :obj:`float`, optional
        Time sampling

    Returns
    -------
    wav : :obj:`np.array`
        Wavelet

    """
    # Create wavelet
    t = np.arange(-length / 2, (length - dt) / 2, dt)
    wav = (1.0 - 2.0 * (np.pi ** 2) * (fc ** 2) * (t ** 2)) * \
          np.exp(-(np.pi ** 2) * (fc ** 2) * (t ** 2))

    # Apply random polarity
    pol = np.random.choice([-1, 1])
    wav *= pol
    return wav


def bandpassed_noise(fmin, fmax, nf=1024, fs=1):
    """Band-passed noise

    Generate band-passed random noise

    Parameters
    ----------
    fmin : :obj:`float`
        Minimum frequency
    fmax : :obj:`float`
        Maximum frequency
    nf : :obj:`float`
        Number of positive frequencies
    dt : :obj:`float`, optional
        Time sampling

    Returns
    -------
    n : :obj:`np.array`
        Noise

    """
    freqs = np.abs(np.fft.rfftfreq(nf, 1. / fs))
    # define frequency mask
    f = np.zeros(nf, dtype='complex')
    idx = np.where(np.logical_and(freqs >= fmin, freqs <= fmax))[0]
    f[idx] = 1
    # create noise spectrum with random phase and unitary amplitude
    phases = np.random.rand(len(f)) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f *= phases
    # compute noise in time domain
    n = np.fft.irfft(f)
    return n


def make_synthetic(wav, nt=500):
    """Synthetic data

    Create synthetic data composed of one event

    Parameters
    ----------
    wav : :obj:`np.array`
        Wavelet
    nt : :obj:`int`, optional
        Number of time steps

    Returns
    -------
    data : :obj:`np.array`
        Data
    ievent : :obj:`int`
        Index of event

    """
    data = np.zeros(nt)

    # identify index of event
    win = int(nt / 4) # do not place event towards the edges of time axis
    ievent = np.random.randint(win, nt - win)

    # create data
    data[ievent] = 1
    data = np.convolve(data, wav, 'same')
    data = data / max(abs(data))
    return data, ievent


def compute_labels(signal, thresh=1e-3):
    """Compute labels

    Define label data based on threshold on input signal

    Parameters
    ----------
    signal : :obj:`np.array`
        Input signal
    thesh : :obj:`float`, optional
        Threshold

    Returns
    -------
    labels : :obj:`np.array`
        Labels

    """
    labels = np.zeros(len(signal))
    labels[np.where(abs(signal) > thresh)] = 1
    return labels


def create_data(nt=500, dt=0.002,
                snrparams=[0.1, 3., 8, 0.3],
                freqwav=[10, 20],
                freqbp=[2, 25],
                signal=True):
    """Data creation

    Create trace with single event with additive band-pass noise

    Parameters
    ----------
    nt : :obj:`int`, optional
        Number of time steps
    dt : :obj:`float`, optional
        Time sampling
    snrparams : :obj:`list`, optional
        Signal-to-noise ratio distribution parameters
        (see `draw_snr` for details)
    freqwav : :obj:`list`, optional
        Range of frequencies where the wavelet central frequency is
        randomly selected
    freqbp : :obj:`list`, optional
        Low- and high-cut frequencies used to filter the data at the end of
        the generation process

    Returns
    -------
    dictdata : :obj:`dict`
        Dictionary containing data and labels (plus other intermediate results)

    """

    # 1) wavelet creation
    fc = np.random.randint(freqwav[0], freqwav[1])
    wavelet = ricker(fc=fc, dt=dt)

    # 2) create data & label
    event, ievent = make_synthetic(wavelet, nt=nt)

    # 3) create noise
    snr = draw_snr(*snrparams)
    noise = bandpassed_noise(fmin=2, fmax=120,
                             nf=2**math.floor(math.log2(nt))+1,
                             fs=1./dt)[:nt]
    
    # 4) Scale traces
    n_rms = np.sqrt(np.mean(noise ** 2))
    scale_factor = snr * n_rms
    sc_event = event * scale_factor

    # 5) combine signal and noise then generate labels
    if signal:
        synth = noise + sc_event
        labels = compute_labels(event)
    else:
        synth = noise
        labels = np.zeros_like(noise)

    # 6) apply band-pass filter
    synth = butter_bandpass_filter(np.concatenate([noise, synth, noise]),
                                   freqbp[0], freqbp[1], fs=1/dt)[len(noise):-len(noise)]

    # 7) normalize synthetic data
    synth = synth / max(abs(synth))

    # 8) structure data to be returned
    dictdata = {'synthetic': synth,
                'event': event,
                'noise': noise,
                'labels': labels,
                'wavelet': wavelet,
                'snr': snr,
                't': np.arange(nt) * dt,
                'has_signal': signal,
                'peak_arrival': ievent
                }

    return dictdata

