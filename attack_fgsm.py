import numpy as np
import torch
from attack_dataloader import AttackaudioDataLoader 
from attack_dataloader import AttackspectrogramDataset

from warpctc_pytorch import CTCLoss
from decoder import GreedyDecoder , BeamCTCDecoder
from model import DeepSpeech, supported_rnns
#from data.utils import reduce_tensor

import scipy.io.wavfile as wav
from PGD_attack_dataloader import SpectrogramDataset,SpectrogramDataLoader

import torch.nn as nn
import json
from tqdm import tqdm
import time
class PGDAttack:
    def __init__(self,model,audio_conf,device,adv_lr,optimizer_type = 'Adam',momentum = 0.9):
        
        self.device = device
        self.optimizer_type =  optimizer_type
        self.adv_lr = adv_lr
        self.momentum = momentum
        self.model = model
        self.audio_conf = audio_conf
        #with open(labels_path) as label_file:
           #self.labels = str(''.join(json.load(label_file)).lower())
           
        self.model = self.model.to(device)
        self.model.train()

        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = False
        for m in self.model.modules():
            if isinstance(m,nn.BatchNorm1d) or isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.Dropout):
                m.eval()

        self.criterion = CTCLoss()

    def attack(self,audios, audio_lengths,targets,target_sizes,epsilon):
        print('starting attack')
        original = torch.clone(audios)
        batch_size = audios.size(0)
        
        final_audio = torch.zeros(original.shape)
    
        delta = torch.FloatTensor((audios.shape)).uniform_(-epsilon,epsilon).to(self.device)
        delta.requires_grad= True
        
        mask = torch.ones((batch_size,torch.max(audio_lengths).item()), device = self.device)
            
        for l , k in enumerate(audio_lengths):
            for i in range(torch.max(audio_lengths).item()-1,-1,-1):
                if (i >= k.item()):
                    mask[l][i] = 0
                else:
                    pass
        
        new_input = delta * mask + original

        if self.optimizer_type is not 'Adam':
            optimizer = torch.optim.SGD([delta],lr = self.adv_lr)
        else:
            optimizer = torch.optim.Adam([delta], lr = self.adv_lr)
        iteration = i
        now = time.time()
                
        spectrograms = SpectrogramDataset(audios = new_input ,  audio_conf = self.audio_conf, normalize =True)

        spectrogram_batch = SpectrogramDataLoader(spectrograms, batch_size)
        for _,(data) in enumerate(spectrogram_batch):
            inputs , input_percentages = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
                
                
        inputs = inputs.to(self.device)

        logits, output_sizes = self.model(inputs, input_sizes)
        logits = logits.transpose(0,1).contiguous()

        loss = self.criterion(logits,targets, output_sizes, target_sizes).to(self.device)
        loss = loss / inputs.size(0)
        
        optimizer.zero_grad()
        
        loss.backward()
        delta.grad = torch.sign(delta.grad*-1)
        optimizer.step()
        
        new_input = delta * mask + original
                        
        new_input = torch.max(torch.min(new_input , original + epsilon),original -epsilon)
        
        noise = torch.from_numpy(np.random.normal(loc =0, scale=2,size= new_input.shape)).type(torch.float32).to(self.device)
        final_audio = torch.clamp(new_input+noise , min = -2**15 ,max = 2**15-1)
        final_audio =  final_audio.round()

        return final_audio
