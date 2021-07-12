import torch
import numpy as np


from data.spectrogram import Spectrogram
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



def load_audio(path):
    sound, _ = torchaudio.load(path, normalization=True)
    sound = torch.transpose(sound,0,1)
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound


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



class AttackspectrogramParser(AudioParser):
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
        y = sound[0]
        # STFT
        D = Spectrogram(self.audio_conf)
        spect = D.__call__(y)
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


class AttackspectrogramDataset(Dataset, AttackspectrogramParser):
    def __init__(self, audio_conf, audios, target, labels, normalize=False):
        """
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        
        """
        
        self.audios = audios
        self.size = len(self.audios)
        self.target = target
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(AttackspectrogramDataset, self).__init__(audio_conf, normalize)

    def __getitem__(self, index):
        sound = self.audios[index]
        transcript = self.target
        spect = self.parse_audio(sound)
        transcript = self.parse_transcript(transcript)
        return spect, transcript

    def parse_transcript(self, transcript):
        transcript = transcript.strip().lower()
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript  ##rank of the transcript_values in the label_map, a : 2 , c : 4 etc dictionary

    def __len__(self):
        return self.size

def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor) #copying the tensor to input(tensors of zeroes).
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes

class AttackaudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AttackaudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn




    




