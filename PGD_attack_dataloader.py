import numpy as np
import torch
import argparse
import os
import scipy.io.wavfile as wav

import torch.nn as nn
from data.spectrogram import Spectrogram
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
 

class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError

class AudiofileDataset_cut(Dataset):
    def __init__(self,audios,ratio):
        self.audios = audios
        self.size = audios.size(0)
        self.ratio = ratio

    def __getitem__(self, index):
        sample = self.audios[index]
        audio, length = sample, len(sample)
        audio_cut = audio[:int(self.ratio*length)]
        length_cut = len(audio_cut)
        return audio_cut, length_cut
    
    def __len__(self):
        return self.size

def _collate_fn_cut(batch):
    def func(p):
        return p[1]
    batch = sorted(batch, key=lambda sample: sample[1], reverse=True)
    #print(batch)
    #print('****************')
    #print(batch[0][1])
    longest_sample = batch[0][0]
    minibatch_size = len(batch)
    max_seqlength = batch[0][1]
    audios = torch.zeros(minibatch_size,max_seqlength)
    audio_len = torch.zeros(minibatch_size)
    #print(inputs.shape)
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        seq_len = sample[1]
        audios[x].narrow(0,0,seq_len).copy_(tensor)
        audio_len[x] = seq_len
    audio_len = audio_len.type(torch.int32)
    return audios , audio_len

class NaturalAudioDataLoader_cut(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(NaturalAudioDataLoader_cut, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_cut

class AudiofileDataset(Dataset):
    def __init__(self, audio_conf, manifest_filepath, labels, normalize=False, augment=False):
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        #super(AudiofileDataset, self).__init__(audio_conf, normalize, augment)
    
    def parse_audio(self,audio_path):
        fs,audio = wav.read(audio_path)
        audio = torch.FloatTensor(audio)
        return audio
    
    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        audio = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path)
        return audio, len(audio), transcript

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '').lower()
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript  ##rank of the transcript_values in the label_map, a : 2 , c : 4 etc dictionary

    def __len__(self):
        return self.size

def _collate_fn(batch):
    def func(p):
        return p[1]
    batch = sorted(batch, key=lambda sample: sample[1], reverse=True)
    #print(batch)
    #print('****************')
    longest_sample = batch[0][0]
    minibatch_size = len(batch)
    max_seqlength = batch[0][1]
    audios = torch.zeros(minibatch_size,max_seqlength)
    audio_len = torch.zeros(minibatch_size)
    #input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    #print(inputs.shape)
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[2]
        seq_len = sample[1]
        audios[x].narrow(0,0,seq_len).copy_(tensor)
        target_sizes[x] = len(target)
        audio_len[x] = seq_len
        targets.extend(target)
    targets = torch.IntTensor(targets)
    audio_len = audio_len.type(torch.int32)
    return audios , audio_len, targets , target_sizes


class NaturalAudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(NaturalAudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        self.normalize = normalize
        self.audio_conf = audio_conf

    def parse_audio(self, sound):
        audio = sound
        # STFT
        D = Spectrogram(self.audio_conf)
        spect = D.__call__(audio)
        # S = log(S+1)
        spect = torch.log1p(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect = spect-mean
            spect = spect/std

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class SpectrogramDataset(Dataset,SpectrogramParser):
    def __init__(self, audio_conf, audios,normalize=False):
        #audios = object of NaturalaudioDataloader class [0] th element
        self.audios = audios
        self.size = self.audios.size(0)
        super(SpectrogramDataset, self).__init__(audio_conf, normalize)

    def __getitem__(self, index):
        #print(self.audios[index])
        sound = self.audios[index]
        spect = self.parse_audio(sound)
        return spect

    def __len__(self):
        return self.size

#this needs to be called in every iteration to create spectrogram of the batch
def _collate2_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample.size(1), reverse=True)
    #longest_sample = max(batch, key=func)[0]
    freq_size = batch[0].size(0)
    minibatch_size = len(batch)
    max_seqlength = batch[0].size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    for x in range(minibatch_size):
        tensor = batch[x]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor) #copying the tensor to input(tensors of zeroes).
        input_percentages[x] = seq_length / float(max_seqlength)
    return inputs,input_percentages

class SpectrogramDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(SpectrogramDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate2_fn
    
    
    
    
