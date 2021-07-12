from __future__ import division, print_function
from warnings import warn
import numpy as np
import torch


class Spectrogram(object):
    def __init__(self, audio_conf, ws=None, 
                 pad=0, window=torch.hamming_window,
                 power=2, normalize=False, wkwargs=None):
        self.sample_rate = audio_conf['sample_rate']
        self.window_size = audio_conf['window_size']
        self.window_stride = audio_conf['window_stride']
        self.n_fft = int(self.sample_rate * self.window_size)
        self.hop_length = int(self.sample_rate * self.window_stride)
        # number of fft bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.win_length = int(self.n_fft)
        self.window = window(self.win_length) if wkwargs is None else window(self.win_length, **wkwargs)
        self.pad = pad
        self.power = power
        self.normalize = normalize
        self.wkwargs = wkwargs

    def __call__(self, sig):
        if self.pad > 0:
            with torch.no_grad():
                sig = torch.nn.functional.pad(sig, (self.pad, self.pad), "constant")
        self.window = self.window.to(sig.device)
        
        

        # default values are consistent with librosa.core.spectrum._spectrogram
        spec_f = torch.stft(sig, n_fft = self.n_fft, win_length = self.win_length, hop_length = self.hop_length,
                            window = self.window, center=True,
                            normalized=False, onesided=True,
                            pad_mode='reflect')
        #print(spec_f.grad)
        if self.normalize:
            spec_f = spec_f/self.window.pow(2).sum().sqrt()
        spec_f = spec_f.pow(self.power).sum(-1)  # get power of "complex" tensor (c, l, n_fft)
        return spec_f
