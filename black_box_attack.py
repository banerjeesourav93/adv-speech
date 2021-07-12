import torch
import argparse
import json
import os
import time
from tqdm import tqdm

from model import DeepSpeech
import torch.nn.functional as F
from attack_madry_inference import PGDAttack

from PGD_attack_dataloader import NaturalAudioDataLoader, AudiofileDataset,SpectrogramDataset,SpectrogramDataLoader
from data.utils import reduce_tensor
from decoder import BeamCTCDecoder


from early_stopping import EarlyStopping

from opts import add_decoder_args

parser = argparse.ArgumentParser(description='Black box attack')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/black_box.csv')
parser.add_argument('--eps', default=2000, type=float, help='Value of l infinity norm bound')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--generator_model_path', default = 'models/deepspeech_final.pth',help='Location to model to generate attack files')
parser.add_argument('--defense_model_path',default = 'defense_models/deepspeech_final_normal.pth'
                            ,help='Location to model to attack')
parser.add_argument('--num-workers', default=1, type=int, help='Number of workers used in data-loading')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum, will be used only if the optimizer is SGD otherwise redundant')
parser = add_decoder_args(parser)

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    

    print("Loading generator model %s" % args.generator_model_path)
    gen_package = torch.load(args.generator_model_path, map_location=lambda storage, loc: storage)
    generator_model = DeepSpeech.load_model_package(gen_package)
    labels = DeepSpeech.get_labels(generator_model)
    audio_conf = DeepSpeech.get_audio_conf(generator_model)
    
    print("Loading defense model %s" % args.defense_model_path)
    def_package = torch.load(args.defense_model_path, map_location=lambda storage, loc: storage)
    def_model = DeepSpeech.load_model_package(def_package)
    
     
    decoder = BeamCTCDecoder(labels,lm_path = args.lm_path,alpha = args.alpha,beta = args.beta,cutoff_top_n = args.cutoff_top_n,cutoff_prob = args.cutoff_prob,beam_width = args.beam_width,num_processes = args.lm_workers)
    
    test_dataset = AudiofileDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest, labels=labels,
                                       normalize=True, augment=False)
    test_loader =  NaturalAudioDataLoader(test_dataset, batch_size=args.batch_size,
                                              num_workers=args.num_workers)
    
    generator_model = generator_model.to(device)
    def_model = def_model.to(device)
    generator_model.train()
    def_model.eval()
    adv_lr_list = [1.1,1.5]
    adv_iterations_list = [50,70,90]
    
   
    for adv_iterations in adv_iterations_list:
        for adv_lr in adv_lr_list:
            total_cer, total_wer = 0, 0
            print("adversarial iteration :{} \t adversarial learning rate :{}".format(adv_iterations,adv_lr))
            attack = PGDAttack(model = generator_model,audio_conf = audio_conf , device =  device , iterations = adv_iterations,optimizer_type = 'dam', adv_lr = adv_lr , momentum = args.momentum)
    
            for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
                audios , audio_lengths, targets , target_sizes = data
                audios = audios.to(device)
                adv_audios = attack.attack(audios = audios , audio_lengths = audio_lengths ,targets = targets , target_sizes = target_sizes, epsilon = args.eps)
        
                adv_audios = adv_audios.detach()
                with torch.no_grad():
                    spectrograms = SpectrogramDataset(audio_conf=audio_conf,audios=adv_audios,normalize = True)
                    adv_spectrogram_batch = SpectrogramDataLoader(spectrograms,args.batch_size)
                    len(adv_spectrogram_batch)
                    for _,(data) in enumerate(adv_spectrogram_batch):
                        inputs , input_percentages = data
                        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            
                    #print(inputs.shape)
                    # unflatten targets
                    split_targets = []
                    offset = 0
                    for size in target_sizes:
                        split_targets.append(targets[offset:offset + size])
                        offset += size
                    inputs = inputs.to(device)

                    out, output_sizes = def_model(inputs, input_sizes)
                    #out = F.softmax(out, dim=-1)
                    decoded_output, _ = decoder.decode(out, output_sizes) #output shape should be (batch_x,seq_x,label_size)
                    target_strings = decoder.convert_to_transcripts(split_targets)
                    wer, cer = 0, 0
                    for x in range(len(target_strings)):
                        transcript, reference = decoded_output[x][0], target_strings[x][0]
                        #print('{} ******** {}'.format(transcript,reference))
                        wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                        cer += decoder.cer(transcript, reference) / float(len(reference))
                    total_cer += cer
                    total_wer += wer
                    del out
                    #print(total_cer , total_wer)
            wer = total_wer / len(test_loader.dataset)
            cer = total_cer / len(test_loader.dataset)
            wer *= 100
            cer *= 100
            print('Validation Summary\t'
                  'Average WER {wer:.3f}\t'
                  'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
            torch.cuda.empty_cache()

        
