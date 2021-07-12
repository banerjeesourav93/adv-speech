import numpy as np
import torch
import argparse
import os
import scipy.io.wavfile as wav
 

from attack_utils import Attack

parser = argparse.ArgumentParser(description="Speech Attack")
#parser.add_argument('--in', type=str, dest="input", nargs='+',
                        #required=True,
                        #help="Input audio .wav file(s), at 16KHz (separated by spaces)")
#parser.add_argument('--target', type=str,
                        #required=True,
                        #help="Target transcription")
parser.add_argument('--manifest_filepath', type =str, default ='data/CW_test.csv', help ='Manifest filepath containing source target pairs')
parser.add_argument('--out',type=str, nargs='+',
                        required=True,
                        help="Path for the adversarial example(s)")
parser.add_argument('--final_path', type=str, default = 'final_attack_files'
                        ,nargs ='+',
                        help="final path for adversarial examples")
parser.add_argument('--finetune', type=str, nargs='+',
                        required=False,
                        help="Initial .wav file(s) to use as a starting point")
parser.add_argument('--lr', type=float,
                        required=False, default=0.5,
                        help="Learning rate for optimization")
parser.add_argument('--momentum', type=int,required=False, default=0.9,help="Learning rate for optimization")

parser.add_argument('--iterations', type=int,
                        required=False, default=1000,
                        help="Maximum number of iterations of gradient descent")
parser.add_argument('--l2penalty', type=float,
                        required=False, default=float('inf'),
                        help="Weight for l2 penalty on loss function")
parser.add_argument('--mp3', action="store_const", const=True,
                        required=False,
                        help="Generate MP3 compression resistant adversarial examples")
parser.add_argument('--decoder_type',default = 'Beam' ,choices=["Greedy", "Beam"],type = str, help ="Decoder type to be used")
parser.add_argument('--optimizer_type',default = 'Adam' ,choices=["Adam", "SGD"],type = str, help ="optimizer type to be used")
parser.add_argument('--cuda', dest='cuda', default = True, action='store_true', help='Use cuda to train model')
parser.add_argument('--model_path', type=str,required=True,help="Best model path to test the attack")
from opts import add_decoder_args

if __name__ == '__main__':
    parser = add_decoder_args(parser)
    args = parser.parse_args()

    finetune = []
    audios = []
    lengths = []
    targets =[]
    #assert args.out is not None
    #assert len(args.input) == len(args.out)
    #if args.finetune is not None and len(args.finetune):
        #assert len(args.input) == len(args.finetune)
        
    # Load the inputs that we're given
    with open(args.manifest_filepath) as f:
        ids = f.readlines()
    ids = [x.strip().split(',') for x in ids]
    #print(ids)
    for i in range(len(ids)-1):
        print(i)
        fs,audio = wav.read(ids[i][1])
        audio = torch.FloatTensor(audio)
        with open(ids[i][2], 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '').lower()
        assert fs == 16000
        print('source dB', 20*np.log10(torch.max(audio)).item())
        audios.append([audio])
        lengths.append(len(audio))
        targets.append([transcript])
    #print(audios)
    #print(lengths)
    #print(targets)
    #def maxleng(audio):
        #return audio[0].size(0)

    #if args.finetune is not None:
        #finetune.append(list(torchaudio.load(args.finetune[i])[1]))

    #maxlen = max(map(maxleng,audios))
    #audios = [[torch.cat((x[0],torch.tensor([0]*(maxlen-len(x[0])),dtype =torch.float)))] for x in audios]
    #finetune = [torch.cat((x[0],torch.tensor([0]*(maxleng-len(x[0])),dtype =torch.float))) for x in finetune]

    
    
    device = torch.device("cuda" if args.cuda else "cpu")
    count =0
    for i, audio in enumerate(audios): 
        sound = [audio]
        length = [lengths[i]]
        sound[0][0] = sound[0][0].to(device)
        target = targets[i][0]
        if args.decoder_type is 'Greedy':
            attack = Attack(model_path = args.model_path,audio_len = length,audios = sound, target = target,optimizer_type = args.optimizer_type, device = device, l2_penalty =args.l2penalty,momentum = args.momentum, lr = args.lr, iterations = args.iterations)
            
            #final_sound = attack.attack(args.out[i])
        else:
            attack  = Attack(model_path = args.model_path,audio_len = length,audios = sound,target = target,device = device,l2penalty = args.l2penalty,iterations = args.iterations ,decoder_type = args.decoder_type,lm_path = args.lm_path,alpha = args.alpha,beta = args.beta,optimizer_type = args.optimizer_type,momentum = args.momentum,lr =args.lr,cutoff_decoder = args.cutoff_top_n, cutoff_prob = args.cutoff_prob,beam_width = args.beam_width,lm_workers = args.lm_workers)
        print('*******************',args.out)  
        final_sound = attack.attack(args.out)
        try:
            print('final_distortion is {0:.3f} decibels'.format(20*torch.log10(torch.mean(torch.abs(sound[0][0] - final_sound[0][0].type(torch.float))))))
            count+= 1
            #wav.write(final_path+'/audio{}.wav'.format(i), 16000,np.array(final_audio[0][0]))
        except:
            print('did not get any audio file_{i}'.format(i=i))
            pass
    print('total audio files attack success is {}'.format(count))


