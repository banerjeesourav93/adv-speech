import numpy as np
import torch
from attack_dataloader import AttackaudioDataLoader 
from attack_dataloader import AttackspectrogramDataset

from tqdm import tqdm
from warpctc_pytorch import CTCLoss
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns

from decoder import GreedyDecoder , BeamCTCDecoder
#from data.utils import reduce_tensor
import torch.nn as nn
import torch.nn.functional as F

import scipy.io.wavfile as wav
import json
import os
import time

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

def get_logits(model,new_audio,audio_conf,labels,target,device,normalize = True):
    attack_dataset = AttackspectrogramDataset(audio_conf=audio_conf,audios= new_audio,labels=labels,target=target,normalize=normalize)
    attack_loader = AttackaudioDataLoader(attack_dataset,len(new_audio))
    inputs,targets,input_percentages,target_sizes= list(attack_loader)[0]
    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
    inputs = inputs.to(device)
    logits,output_sizes = model(inputs,input_sizes)    
    return logits,targets,output_sizes,target_sizes

#def check_list(list_, target):
    #flag = 0
    #for i in range(len(list_)):
        #if list_[i] == target:
            #flag += 1
    #if flag ==0:
        #return False
    #else:
        #return True
    

class Attack:
    def __init__(self,model_path,audio_len,audios,target,device,l2penalty,lm_workers,lr,momentum,optimizer_type = 'Adam',iterations = 1000,decoder_type = 'Greedy',lm_path = None,alpha = 0,beta = 0,cutoff_decoder = 0,cutoff_prob = 0,beam_width = 0):
        self.path = model_path
        self.audio_len = audio_len
        self.audios = audios
        self.device = device
        self.target = target
        self.batch_size = len(self.audios)
        self.phrase_len = len(self.target)
        self.l2penalty = l2penalty
        self.iterations = iterations
        self.momentum = momentum
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.mask = torch.tensor([1 if i < l else 0 for i in range(audio_len[0]) for l in audio_len],dtype = torch.float,device = device)
        p = []
        for i in range(len(self.audios)):
            p.append(audios[i][0])

        self.original = p[0].clone() #this is a one dimensional tensor
        self.rescale = torch.ones(1,dtype = torch.float,device=device)
        
        #print(self.rescale)
        print(self.original)
        print(target)
                
        self.package = torch.load(self.path, map_location=lambda storage, loc: storage)
        self.model = DeepSpeech.load_model_package(self.package)
        self.labels = DeepSpeech.get_labels(self.model)
        self.audio_conf = DeepSpeech.get_audio_conf(self.model)
        self.parameters = self.model.parameters()
        self.model = self.model.to(device)
        self.model.train()

        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = False
        for m in self.model.modules(): 
            if isinstance(m,nn.BatchNorm1d) or isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.Dropout): 
                m.eval()

        self.criterion = CTCLoss()

        if decoder_type == 'Beam':
            self.decoder = BeamCTCDecoder(self.labels,lm_path = lm_path,alpha = alpha,beta = beta,cutoff_top_n = cutoff_decoder,cutoff_prob = cutoff_prob,beam_width = beam_width,num_processes = lm_workers)
        else:
            self.decoder = GreedyDecoder(self.labels)
        



    def attack(self,example_path):
        loss_val = torch.tensor([])
        final_audio = [None]*self.batch_size
        delta = torch.randn((self.audio_len),dtype = torch.float,requires_grad = True,device = self.device)
        if self.optimizer_type is not 'Adam':
            optimizer = torch.optim.SGD([delta], lr=self.lr,momentum=self.momentum, nesterov=True)
        else:
            optimizer = torch.optim.Adam([delta], lr =self.lr)
        now = time.time()
        MAX = self.iterations
        count =0
        for i in tqdm(range(MAX)):
             
            #print('iteration {} of {}'.format(i+1,MAX))
            iteration = i
            now = time.time()
            apply_delta = torch.clamp(delta,min = -200,max = 200)*self.rescale
            new_input = apply_delta * self.mask + self.original
            new_input = [[new_input]]
            noise = torch.from_numpy(np.random.normal(loc = 0, scale = 2, size = new_input[0][0].shape)).type(torch.float32).to(self.device)
            sound = [[torch.clamp(new_input[0][0]+noise , min = -2**15 ,max = 2**15-1)]]
            logits,targets,output_sizes,target_sizes = get_logits(model=self.model,new_audio=sound,audio_conf=self.audio_conf,labels=self.labels,target=self.target,device = self.device)         

            logits = logits.transpose(0, 1)
            ctcloss = self.criterion(logits, targets, output_sizes,target_sizes).to(self.device)
            if not np.isinf(self.l2penalty): 
                loss = (10**-6)*torch.norm((self.original-new_input[0][0]),p=2)**2 + self.l2penalty*ctcloss
                #loss = torch.norm((self.original-self.new_input),p=2)**2 + self.l2penalty*ctcloss
            else :
                loss = ctcloss
            #print('The l2 norm is {l2} and ctc loss is {ctc} , total loss is {total}'.format(l2 = torch.norm((self.original-new_input[0][0]),p=2).item(), ctc = ctcloss.item(), total= loss.item()))
            
            loss = loss/self.batch_size
            temp = loss.detach().cpu()
            loss_val = torch.cat((loss_val,temp))
            optimizer.zero_grad()
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(self.delta, 400)
                 
            optimizer.step()
            
            
            if i%10 == 0 and i>0:
                logits = logits.transpose(0,1)
                logits = F.softmax(logits, dim=-1)
                self.decoded_output,_ = self.decoder.decode(logits,output_sizes)
                #print('The top decoded output is {}'.format(self.decoded_output[0][0])) 
                for k in range(self.batch_size):  
                    transcript = self.decoded_output[k][0] #!!!!!!!!!!!check the dimension of decoded_output once
                    w = self.decoder.wer(transcript, self.target) / float(len(self.target.split()))
                    c = self.decoder.cer(transcript,self.target)/float(len(self.target))
                
                
                
                for batch in range(self.batch_size):
                    if( i%100 == 0  and self.decoded_output[batch][0]==self.target) or (i == MAX-1 and final_audio[batch] is None):
                    #limit = torch.max(torch.abs(self.delta),1,keepdim = True)[0][batch]
                        limit = torch.max(torch.abs(delta)).item()
                        
                        if self.rescale[batch]*200 > limit:
                            print('We have already found a distortion less than {} for audio {}'.format(limit,batch+1))
                            print('reducing the distortion limit')
                            self.rescale[batch] = limit/200.0
                        self.rescale[batch] *= 0.9
                        #final_audio[batch] = [self.new_input[batch][0]]
                        final_audio[batch] = [new_input[batch][0]]

                        print("Audio_file {} of {} , loss = {} , bound = {}" .format(batch,self.batch_size-1, loss.item() ,200*self.rescale[batch].item())) 
                        final_audio[batch][0] = final_audio[batch][0].squeeze().type(torch.int16)   
                        print('The average distortion added in decibels of {iteration}th iteration is {0:.3f}'.format(20*(torch.log10(torch.max(delta)).item() - torch.log10(torch.max(self.original)).item()) , iteration =i))
                        print('saving the new audio file to disk')
                        count +=1
                        #wav.write(example_path[0] + '/iteration_{}.wav'.format(i), 16000,np.array(final_audio[0][0].cpu()))
            del loss
        #torch.save(loss_val,'/carlini_wagner_loss_values') 
        print('total adv files obtained for this audio is {}'.format(count))
        return final_audio
