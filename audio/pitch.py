import librosa 
import numpy as np
import torch 
import pyworld as pw

# https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/examples/speech_synthesis/evaluation/eval_f0.py#L87
def yin(sig, config):

    '''Compute the Yin Algorithm. Return fundamental frequency and harmonic rate.
    An adaptation of https://github.com/NVIDIA/mellotron which is an adaption of
    https://github.com/patriceguyot/Yin

        Args:
            sig (list of floats): Audio Signal
            sr (int): Sample Rate
            win_len (int): Size of the analysis window (samples)
            hop_length (int): Size of the lag between two consecutives windows (samples)
            f0_min (int): Minimum fundamental frequency that can be detected (hertz)
            f0_max (int): Maximum fundamental frequency that can be detected (hertz)
            harmo_thresh (float): Threshold of detection. The algorithm returns the
            first minimum of the CMND function below this threshold.

        Returns:
            (dict) with:
                'pitches' (1-D np.array): fundamental frequencies,
                'harmonic_rates' (1-D np.array): list of harmonic rate values for each
                    fundamental frequency value (= confidence value)
                'argmins' (1-D np.array): minimums of the Cumulative Mean Normalized 
                    DifferenceFunction
                'times' (1-D np.array): list of time of each estimation
    '''
    sr = config.sr
    win_length = config.win_length
    hop_length = config.hop_length
    f0_min = config.f0_min_pitch
    f0_max = config.f0_max_pitch
    harmo_thresh = config.harmo_thresh

    tau_min = int(sr / f0_max)
    tau_max = int(sr / f0_min)

    # Time values for each analysis window
    time_scale = range(0, len(sig) - win_length, hop_length)
    times = [t/float(sr) for t in time_scale]
    frames = [sig[t:t + win_length] for t in time_scale]

    pitches = [0.0] * len(time_scale)
    harmonic_rates = [0.0] * len(time_scale)
    argmins = [0.0] * len(time_scale)

    for i, frame in enumerate(frames):
        # Compute YIN
        df = _difference_function(frame, win_length, tau_max)
        cm_df = _cumulative_mean_normalized_difference_function(df, tau_max)
        p = _get_pitch(cm_df, tau_min, tau_max, harmo_thresh)

        # Get results
        if np.argmin(cm_df) > tau_min:
            argmins[i] = float(sr / np.argmin(cm_df))
        if p != 0:  # A pitch was found
            pitches[i] = float(sr / p)
            harmonic_rates[i] = cm_df[p]
        else:  # No pitch, but we compute a value of the harmonic rate
            harmonic_rates[i] = min(cm_df)

    return {'pitches' : np.array(pitches), 'harmonic_rates' : np.array(harmonic_rates), 'argmins' : np.array(argmins), 'times' : np.array(times)}

def _difference_function(x, n, tau_max):

    '''Compute difference function of data x. This solution is implemented directly
    with Numpy fft.
    
        Args:
            x (list): Audio data frame
            n (int): Length of data
            tau_max (int): Integration window size
            
        Returns:
            (list) of difference function
    '''
    x = np.array(x, np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)
    fc = np.fft.rfft(x, size_pad)
    conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
    return x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - \
        2 * conv

def _cumulative_mean_normalized_difference_function(df, n):
    
    '''Compute cumulative mean normalized difference function (CMND).
    
        Args:
            df (list): Difference function
            n (int): Length of data
        
        Returns:
            (list) of cumulative mean normalized difference function
    '''

    # Scipy method
    cmn_df = df[1:] * range(1, n) / np.cumsum(df[1:]).astype(float)
    return np.insert(cmn_df, 0, 1)

def _get_pitch(cmdf, tau_min, tau_max, harmo_th=0.1):

    '''Return fundamental period of a frame based on CMND function.
    
        Args:
            cmdf (list): Cumulative Mean Normalized Difference function
            tau_min (int): Minimum period for speech
            tau_max (int): Maximum period for speech
            harmo_th (float): harmonicity threshold to determine if 
            it is necessary to compute pitch frequency

        Returns:
            (float) fundamental period if there is values under threshold, 
            0 otherwise
    '''

    tau = tau_min
    while tau < tau_max:
        if cmdf[tau] < harmo_th:
            while tau + 1 < tau_max and cmdf[tau + 1] < cmdf[tau]:
                tau += 1
            return tau
        tau += 1

    return 0    # if unvoiced

# https://github.com/coqui-ai/TTS/blob/main/TTS/utils/audio.py#L710
def dio(sig, config):
    '''Compute pitch (f0) of a waveform using the same parameters used for 
    computing melspectrogram.
    
        Args:
            sig (list of floats): Audio Signal
            sr (int): Sample Rate
            hop_length (int): Number of frames between STFT columns
            f0_min (int): Minimum fundamental frequency that can be detected (hertz)
            f0_max (int): Maximum fundamental frequency that can be detected (hertz)
    
        Returns:
            (dict) with:
                'pitches' (1-D np.array): fundamental frequencies,
                'times' (1-D np.array): list of time of each estimation
    '''
    sr = config.sr
    hop_length = config.hop_length
    f0_min = config.f0_min_pitch
    f0_max = config.f0_max_pitch


    # Align F0 length to the spectrogram length
    if len(sig) % hop_length == 0:
        sig = np.pad(sig, (0, hop_length // 2), mode="reflect")

    f0, t = pw.dio(
        sig.astype(np.double),
        fs=sr,
        f0_ceil=f0_max,
        f0_floor = f0_min,
        frame_period=1000 * hop_length / sr,
    )
    f0 = pw.stonemask(sig.astype(np.double), f0, t, sr)
    return {'pitches':f0, 'times':t}

# NOT TESTED YET
# https://github.com/NVIDIA/DeepLearningExamples/blob/a43ffd01cb002b23a98c97c3c5a231e24a57fa71/PyTorch/SpeechSynthesis/FastPitch/fastpitch/data_function.py#L81
def compute_pyin(wav, mel_len, method='pyin', normalize_mean=None,normalize_std=None, n_formants=1):

    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)

    if method == 'pyin':

        snd, sr = librosa.load(wav)
        pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
            snd, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), frame_length=1024)
        assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0

        pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
        pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
        pitch_mel = torch.nn.functional.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))

        if n_formants > 1:
            raise NotImplementedError

    else:
        raise ValueError

    pitch_mel = pitch_mel.float()

    if normalize_mean is not None:
        assert normalize_std is not None
        pitch_mel = _normalize_pitch(pitch_mel, normalize_mean, normalize_std)

    return pitch_mel

def _normalize_pitch(pitch, mean, std):
    zeros = (pitch == 0.0)
    pitch -= mean[:, None]
    pitch /= std[:, None]
    pitch[zeros] = 0.0
    return pitch