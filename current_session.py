# coding: utf-8
import argparse
import json
import os
import time

import torch.distributed as dist
import torch.utils.data.distributed
from tqdm import tqdm
from warpctc_pytorch import CTCLoss

from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from data.utils import reduce_tensor
from decoder import GreedyDecoder , BeamCTCDecoder
from model import DeepSpeech, supported_rnns

from opts import add_decoder_args
parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='libri_train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='libri_val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=5, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser = add_decoder_args(parser)

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)
help(torch.cuda.set.device)
import torch
help(torch.cuda.set.device)
torch.cuda.set_device
help(torch.cuda.set_device)
parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='libri_train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='libri_val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=5, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=[0,1],
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser = add_decoder_args(parser)

                    
args = parser.parse_args()
torch.cuda.set_device(int(args.gpu_rank))
torch.cuda.get_device_name(0)
torch.cuda.get_device_name(1)
torch.cuda.device_count()
train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=args.augment)
                                       
audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))

                          
train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
                                   
train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=args.augment)
                                       
with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)).lower())
            
train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=args.augment)
                                       
train_dataset
print(train_dataset)
torch.nn.parallel.DistributedDataParallel
34+150+48+125+156
dist = DistributedBucketingSampler(train_dataset,batch_size=1,rank=2)
dist = DistributedBucketingSampler(train_dataset,batch_size=1,rank=1)
args.world_size
len(train_dataset)
bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
ids = len(train_dataset)
bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
ids
len(ids)
ids = list(range(0, len(data_source)))
ids = list(range(0, len(train_dataset)))
ids
bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size=10)]
bins = [ids[i:i + batch_size] for i in range(0, len(ids),10)]
bins = [ids[i:i + 10] for i in range(0, len(ids),10)]
bins
import math
num_samples = int(math.ceil(len(bins) * 1.0 / 2))
num_samples
len(bins)
num_samples = int(math.ceil(len(bins) * 1.0 / 3))
num_samples
bins
x = bins[:(total_size - len(bins))]
x = bins[:(12 - len(bins))]
x
len(bins)
samples = bins[2::3]
samples
bins
iter(samples)
y = iter(samples)
y
bin_ids = torch.randperm(len(self.bins),generator =g)
bin_ids = torch.randperm(len(bins),generator =g)
g = torch.Generator()
bin_ids = torch.randperm(len(bins),generator =g)
bin_ids
bins = [bins[i] for i in bin_ids]
bins
samples
train_sampler = DistributedBucketingSampler(train_dataset,10,3,2)
train_sampler.shuffle()
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
                    
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
                    
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
                    parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
                    
parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
                    
args = parser.parse_args()
args.no_sorta_grad
parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',default =True
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
                    
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',default =True, help='Turn off ordering of dataset on sequence length for the first epoch.')
                    
args.no_sorta_grad
args = parser.parse_args()
args.no_sorta_grad
parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='libri_train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='libri_val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=5, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=0,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser = add_decoder_args(parser)

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)
args = parser.parse_args()
args.distributed
args.distributed = args.worldsize
args.world_size
args.distributed = args.world_size
args.distributed
args.distributed = args.world_size>1
args.distributed
model = torch.nn.parallel.DistributedDataParallel(model,device_ids=1)
dist = DistributedBucketingSampler(train_dataset,batch_size=1,rank=1)
train_dataset
dist = DistributedBucketingSampler(train_dataset,batch_size=1,rank=1)
dist = DistributedBucketingSampler(train_dataset)
dist = DistributedBucketingSampler(train_dataset,num_replicas=2,batch_size=1,rank=1)
dist
len(dist)
args.num_workers
train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=dist)
                                   
train_loader
for i, (data) in enumerate(train_loader, start=start_iter):     print(i,(data))
for i, (data) in enumerate(train_loader, start=0):     print(i,(data))
for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            print(inputs,targets,input_percentages,target_sizes)
            
for i, (data) in enumerate(train_loader, start=len(train_loader)-3):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            print(inputs,targets,input_percentages,target_sizes)
            
len(train_loader)
for i, (data) in enumerate(train_loader, start=len(train_loader)+20):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            print(inputs,targets,input_percentages,target_sizes)
            
for i, (data) in enumerate(train_loader, start=len(train_loader)+20):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            print(targets,input_percentages,target_sizes)
            
for i, (data) in enumerate(train_loader, start=len(train_loader)+20):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            print(input_percentages,target_sizes)
            
for i, (data) in enumerate(train_loader, start=len(train_loader)+20):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            print(inputs,targets)
            
inputs.size
inputs
inputs.size()
targets.size()
targets.max()
targets.min()
targets.mean()
targets.mode()
targets.median()
enumerate
help(enumerate)
train_loader
dtype(train_loader)
type(train_loader)
my_list = ['apple', 'banana', 'grapes', 'pear']
for c , val in enumerate(my_list)
for c , val in enumerate(my_list):
    print(c,val)
    
for c , val in enumerate(my_list,-1):
    print(c,val)
    
for c , val in enumerate(my_list,2):
    print(c,val)
    
torch.utils.data
help(torch.utils.data)
help(torch.utils.data.dataloader)
inputs
for i, (data) in enumerate(train_loader, start=len(train_l


oa
     ...: der)+20):
     ...:             if i == len(train_sampler):
     ...:                 break
     ...:             inputs, targets, input_percentages, rget_sizes= data            print(inputs,targets)
for i, (data) in enumerate(train_loader, start=len(train_loader)+20):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            print(inputs,targets)
            
batch = sorted (inputs,key = lambda sample: sample[0].size(1),reverse = True)
batch
batch.size()
batch.shape
type(batch)
sample
batch = sorted (inputs,key = lambda sample: sample[0].size(1),reverse = True)
sample[0].size()
a = torch.randn((2,3,4.2),dtype = torch.float)
a = torch.randn((2,3,4,2),dtype = torch.float)
a
a[0]
a[0][1]
a[0,1]
a[0,2]
a[0,3]
a.shape
a[0]
a[0]size(1)
a[0].size(1)
a[0]
a[0].size()
a[0].size(0)
a[0].size(1)
train_dataset.size()
train_dataset
(train_dataset)
print(train_dataset)
get_ipython().run_line_magic('history', '')
(data)
typr(data)
type(data)
data[0]
type(data(0))
max(batch)
longest_sample = max(batch)
longest_sample.size()
longest_sample.size(1)
len(batch)
batch
batch = sorted (inputs,key = lambda sample: sample[0].size(1),reverse = True)
inputs
for i, (data) in enumerate(train_loader, start=len(train_loader)+20):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            print(inputs,targets)
            
batch = sorted (inputs,key = lambda sample: sample[0].size(1),reverse = True)
batch
a[0].size(1)
a[0]
batch.shape()
batch.shape
batch
data
(data)
for i, (data) in enumerate(train_loader, start=len(train_loader)+20):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            print(inputs,targets)
            
inputs
mport os
import subprocess
from tempfile import NamedTemporaryFile

from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler

import librosa
import numpy as np
import scipy.signal
import torch
import torchaudio
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import subprocess
from tempfile import NamedTemporaryFile

from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler

import librosa
import numpy as np
import scipy.signal
import torch
import torchaudio
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
output = subprocess.check_output(['soxi -D \"%s|"'%path.strip()],shell=True)
get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('cd', 'home')
get_ipython().run_line_magic('cd', '..')
pwd = DeepSpeech/data/ldc93s1/LDC93S1.wav
path = DeepSpeech/data/ldc93s1/LDC93S1.wav
path = "DeepSpeech/data/ldc93s1/LDC93S1.wav"
output = subprocess.check_output(['soxi -D \"%s|"'%path.strip()],shell=True)
ped
get_ipython().run_line_magic('pwd', '')
path
path = "/home/sourav/DeepSpeech/data/ldc93s1/LDC93S1.wav"
output = subprocess.check_output(['soxi -D \"%s|"'%path.strip()],shell=True)
get_ipython().run_line_magic('cd', 'dissertation_project/')
output = subprocess.check_output(['soxi -D \"%s|"'%path.strip()],shell=True)
output = subprocess.check_output(['soxi -D \"%s|"'%path.strip()],shell=True)
path.strip()
path = "/home/sourav/DeepSpeech/data/ldc93s1"
output = subprocess.check_output(['soxi -D \"%s|"'%path.strip()],shell=True)
a
a.unsqueeze(1)
a.shape
b = a.unsqueeze(1)
b.shape
b = a.unsqueeze(1)
b.shape
b =b.unsqueeze(1)
b.shape
b =b.unsqueeze(0)
b.shape
b
b.shape
np.random.rand()
np.random.rand()
np.random.rand()
np.random.rand()
np.random.rand()
np.random.rand()
np.random.rand()
np.random.rand()
np.random.rand()
np.random.rand()
np.random.rand()
torch.device
torch.device()
help(torch.device)
torch.device('cuda:0')
torch
help(dist.init_process_group)
import torch.distributed as dist
help(dist.init_process_group)
help(torch.cuda)
help(torch.utils.data.dataset)
help(torch.utils.data.dataloader)
from torch.utils.data import dataloader
c = dataloader()
float('inf')
type(float('inf'))
torch.utils.data
t = torch.utils.data.dataset
help(t)
help(np.random.binomial)
x = torch.FloatTensor(16)
x
y = torch.IntTensor(16)
y
inputs = torch.zeros(8, 1, 15, 20)
inputs
inputs = torch.zeros(3, 1, 3, 5)
inputs
inputs[1]
inputs[1][0]
inputs[3]
inputs[2]
train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=args.augment)
                                       
train_dataset
train_loader = AudioDataLoader(train_dataset,
                                  num_workers=args.num_workers, batch_sampler=train_sampler)
                                 
for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            
train_loader.collate_fn(train_dataset)
batch = sorted(train_dataset,key=lambda sample: sample[0].size(1), reverse=True)
batch
train_dataset[0]
train_dataset
train_dataset[1]
len(train_dataset)
train_dataset.shape
train_dataset.size()
train_dataset[0].size()
longest_sample = max(batch, key=func)[0]
def func(p):
        return p[0].size(1)
longest_sample = max(batch, key=func)[0]
longest_sample
longest_sample[0].size(1)
longest_sample.size(1)
longest_sample = max(batch, key=func)
longest_sample
longest_sample = len(max(batch, key=func)[1])
longest_sample
batch
batch[0]
batch[0].size(1)
batch[0][0]
train_dataset
for temp in train_dataset:
    break
print(temp)
list(train_dataset)
for temp in train_dataset:
    print(temp[0].size(1))
    
for temp1 in batch:
    break
train_dataset.__iter__
dir(train_dataset)
train_dataset.__getitem__
batch[0].shape
batch
batch[0][0]
batch[0][0].shape
type(batch.[0][0])
type(batch[0][0])
batch[0][0].size(1)
batch[1][0].size(1)
batch[2][0].size(1)
for temp in train_dataset:
    print(temp[0].size(1))

        
for x in batch:
    print(x[0].size(1))
    
for x in batch:
    print(x[0])
    
for x in batch:
    print(x[1])
    
    
for x in batch:
    len(x[1])
    
    
    
for x in batch:
   print( len(x[1]))
   
    
    
    
for x in batch:
   print( len(x[0]))
   
   
    
    
    
for x in batch:
   print(x[0].size(1))
   
   
   
    
    
    
longest_sample
longest_sample = max(batch, key=func)[0]
longest_sample
freq_size = longest_sample.size(0)
freq_size
batch[0].shape
batch[0][0].shape
len(batch)
longest_sample.size(1)
freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
freq_size = longest_sample.size(0)
minibatch_size = len(batch)
max_seqlength = longest_sample.size(1)
max_seqlength
inputs = torch.zeros(92, 1, freq_size, max_seqlength)
inputs
input_percentages = torch.FloatTensor(minibatch_size)
input_percentages
target_sizes = torch.IntTensor(minibatch_size)
target_sizes
targets = []
sample = batch[0]
sample
tensor = sample[0]
target = sample[1]
tensor
target
type(tensor)
tensor.size(0)
tensor.size(1)
target.size()
type(target)
list(target)
len(target)
seq_length = tensor.size(1)
seq_length
inputs[0][0].narrow(1, 0, seq_length)
input_percentages
for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        
input_percentages
target_sizes
target_sizes[0]
target_sizes[0].len(target)
target_sizes[0]=len(target)
target_sizes
for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        
target_sizes
for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        print(len(target))
        
        
len(target)
target
targets.extend(target)
targets
for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
        
targets
len(targets)
for x in batch:
    target = batch[x][1]
    y += len(targets)
y = 0
for x in batch:
    target = batch[x][1]
    y += len(targets)
    
for x in batch:
    target = batch[x][1]
    y += len(target)
    
    
target
for x in batch:
    target = batch[x][1]
    print(target)
    
    
    
for x in batch:
    target = batch[x][1]
    print(target)
    
    
    batch[0]
    
batch[0]
batch[1]
for x in batch:
    target = batch[x][1]
    print(target)
    
    
    batch[0][1]
    
    
batch[0][1]
len(batch[0][1])
type(batch[0][1])
for x in batch:
    target_len =len( batch[x][1])
    print(target_len)
    
for x in batch:
    target_len =len( batch[x][1])
    target_len =+ target_len
    print(target_len)
    
    
batch[0][1]
len(batch[0][1])
for x in batch:
    for y in len(minibatch_size):
        target_len =len( x[y][1])
        target_len =+ target_len
        print(target_len)
        
    
    
for x in batch:
    for y in len(minibatch_size):
        target_len =len( x[y][1])
        target_len =+ target_len
        print(target_len)
        
    
    
len(minibatch_size)
minibatch_size
for x in batch:
    for y in (minibatch_size):
        target_len =len( x[y][1])
        target_len =+ target_len
        print(target_len)
        
        
    
    
for x in batch:
    for y in range(minibatch_size):
        target_len =len( x[y][1])
        target_len =+ target_len
        print(target_len)
        
        
        
    
    
for x in batch:
    target_len =len( x[1])
    target_len =+ target_len
    print(target_len)
    
        
        
        
    
    
for x in batch:
    target_len =len( x[1])
    target_len =+ target_len
print(target_len)


    
        
        
        
    
    
target_len=0
for x in batch:
    target_len =len( x[1])
    target_len =+ target_len
print(target_len)


    
        
        
        
    
    
for x in batch:
    t =len( x[1])
    target_len =+ t
print(target_len)


    
        
        
        
    
    
target_len=0
for x in batch:
    t =len( x[1])
    target_len =+ t
print(target_len)


    
        
        
        
    
    
t
for x in batch:
    t =len( x[1])
    target_len =+ t
    print(t)
print(target_len)



    
        
        
        
    
    
for x in batch:
    t =len( x[1])
    target_len =target_len+ t
    print(t)
print(target_len)





    
        
        
        
    
    
for x in batch:
    t =len( x[1])
    target_len += t
    print(t)
print(target_len)







    
        
        
        
    
    
target_len=0
for x in batch:
    t =len( x[1])
    target_len += t
    print(t)
print(target_len)







    
        
        
        
    
    
range(10)
help(torch.narrow)
inputs
sample
tensor.size(1)
type(tensor)
tensor.shape
inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
        
inputs
tensor
type(inputs)
inputs.shape(1)
inputs.size(1)
inputs
inputs[0]
inputs[0].size(1)
inputs[0].size(0)
inputs[0].size(2)
inputs[1].size(2)
inputs[12].size(2)
for i, (data) in enumerate(train_loader, start=0):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            
data
list(data)
len(list(data))
list(data)[0]
len(list(data)[0])
inputs.size(3)
inputs
inputs.size(2)
inputs.size(1)
inputs.size(0)
input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
input_sizes
inputs.size(3)
target_sizes
batch[0][1]
len(batch[0][1])
target_sizes
len(batch[2][1])
len(batch[12][1])
batch[1]
for x in range(minibatch_size):
       sample = batch[x]
       tensor = sample[0]
       target = sample[1]
       seq_length = tensor.size(1)
       inputs[x][0].narrow(1, 0, seq_length).copy_(tensor) #copying the tensor to input(tensors of zeroes).
       input_percentages[x] = seq_length / float(max_seqlength)
       target_sizes[x] = len(target)
       targets.extend(target) for x in range(minibatch_size):
       sample = batch[x]
       tensor = sample[0]
       target = sample[1]
       seq_length = tensor.size(1)
       inputs[x][0].narrow(1, 0, seq_length).copy_(tensor) #copying the tensor to input(tensors of zeroes).
       input_percentages[x] = seq_length / float(max_seqlength)
       target_sizes[x] = len(target)
       targets.extend(target)
for x in range(minibatch_size):
       sample = batch[x]
       tensor = sample[0]
       target = sample[1]
       seq_length = tensor.size(1)
       inputs[x][0].narrow(1, 0, seq_length).copy_(tensor) #copying the tensor to input(tensors of zeroes).
       input_percentages[x] = seq_length / float(max_seqlength)
       target_sizes[x] = len(target)
       targets.extend(target)
      
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
    
target_sizes
targets
get_ipython().run_line_magic('save', 'current_session ~0/')
