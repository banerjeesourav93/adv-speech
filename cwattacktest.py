import numpy as np
import torch
import argparse
import os
import scipy.io.wavfile as wav
from attack_utils import Attack
 
from opts import add_decoder_args

parser = argparse.ArgumentParser(description="Speech Attack")
parser.add_argument('--in', type=str, dest="input", nargs='+',
                        required=True,
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")
parser.add_argument('--target', type=str,
                        required=True,
                        help="Target transcription")
parser.add_argument('--out',type=str, nargs='+',
                        required=True,
                        help="Path for the adversarial example(s)")
parser.add_argument('--final_path', type=str,
                        required=True, nargs ='+',
                        help="final path for adversarial examples")
parser.add_argument('--finetune', type=str, nargs='+',
                        required=False,
                        help="Initial .wav file(s) to use as a starting point")
parser.add_argument('--lr', type=int,
                        required=False, default=100,
                        help="Learning rate for optimization")
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

parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')

parser.add_argument('--model-path', default='models/deepspeech_final_23g.pth',
                    help='Location to save best validation model')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')


if __name__ == '__main__':
    parser = add_decoder_args(parser)
    args = parser.parse_args()

    finetune = []
    audios = []
    lengths = []

    assert args.out is not None
    assert len(args.input) == len(args.out)
    #if args.finetune is not None and len(args.finetune):
        #assert len(args.input) == len(args.finetune)
        
    # Load the inputs that we're given
    for i in range(len(args.input)):
        fs,audio = wav.read(args.input[i])
        audio = torch.FloatTensor(audio)
        assert fs == 16000
        print('source dB', 20*np.log10(torch.max(audio)).item())
        audios.append([audio])
        lengths.append(len(audio))

    #def maxleng(audio):
        #return audio[0].size(0)

    #if args.finetune is not None:
        #finetune.append(list(torchaudio.load(args.finetune[i])[1]))

    #maxlen = max(map(maxleng,audios))
    #audios = [[torch.cat((x[0],torch.tensor([0]*(maxlen-len(x[0])),dtype =torch.float)))] for x in audios]
    #finetune = [torch.cat((x[0],torch.tensor([0]*(maxleng-len(x[0])),dtype =torch.float))) for x in finetune]

    target = args.target

    device = torch.device("cuda" if args.cuda else "cpu")

    for i, audio in enumerate(audios):
        
        sound = [audio]
        length = [lengths[i]]

        if args.decoder_type is 'Greedy':
            attack = Attack(model_path = args.model_path,audio_len = length,audios = sound, target = target, device = device, l2penalty =l2_penalty, iterations = args.iterations[i],lr = args.lr,momentum = args.momentum)
            
            final_sound = attack.attack(args.out[i])
        else:
            attack  = Attack(model_path = args.model_path,audio_len = length,audios = sound,target = target,device = device,l2penalty = args.l2penalty,iterations = args.iterations ,decoder_type = args.decoder_type,lm_path = args.lm_path,alpha = args.alpha,beta = args.beta,cutoff_decoder = args.cutoff_top_n,cuttoff_prob = args.cutoff_prob,beam_width = args.beam_width,lm_workers = args.lm_workers,lr = args.lr,momentum =args.momentum)
    
            final_sound = attack.attack(args.out[i])
        
        print('final_distortion is {0:.3f} decibels'.format(20*torch.log10(torch.mean(torch.abs(sound[0][0] - final_sound[0][0].type(torch.float))))))
        
            
        wav.write(final_path+'/audio{}.wav'.format(i), 16000,np.array(final_audio[0][0]))
