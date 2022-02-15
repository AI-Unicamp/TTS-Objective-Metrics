import torch
import torchaudio

class GlobalConfig():
    def __init__(self):

        self.sr = 22050

        self.f0_min_pitch: int = 100
        self.f0_max_pitch: int = 500

        self.n_fft = 1024
        self.win_length: int = 1024
        self.hop_length: int = 256 
        self.f0_min_mel: int = 0
        self.f0_max_mel = None
        self.harmo_thresh: float = 0.1
        self.window_fn =  torch.hann_window
        self.log_mels = True
        self.n_mels = 80
        self.n_mfcc = 13
        self.power = 2
        self.fig_size: tuple = (16,10)
        self.dist_fn = 'compute_rms_dist'
        self.norm_align_type =  'path'

        self.mel_fn = torchaudio.transforms.MelSpectrogram(
                sample_rate = self.sr, n_fft=self.n_fft, win_length=self.win_length,
                hop_length=self.hop_length, f_min=self.f0_min_mel, f_max = self.f0_max_mel,
                n_mels=self.n_mels, window_fn=self.window_fn, power = self.power
            )

        self.melkwargs = {
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "f_min": self.f0_min_mel,
            "f_max": self.f0_max_mel,
            "window_fn": self.window_fn,
            "n_mels": self.n_mels,
            "power": self.power
        }

        self.mfcc_fn = torchaudio.transforms.MFCC(
                sample_rate = self.sr, n_mfcc=self.n_mfcc, log_mels=self.log_mels, melkwargs=self.melkwargs
            )  
