import sys
import torch
import argparse

import scipy.io.wavfile as wav

from attack_madry import PGDAttack
from PGD_attack_dataloader import NaturalAudioDataLoader, AudiofileDataset, SpectrogramDataset, SpectrogramDataLoader
from model import DeepSpeech
import torch.nn.functional as F
from decoder import BeamCTCDecoder

from opts import add_decoder_args
parser = argparse.ArgumentParser(description='temporal modelling check')
parser.add_argument('--in', type=str, dest="input",
                    required=True,
                    help="Input audio .wav file(s), at 16KHz (separated by spaces)")
parser.add_argument('--adv_iterations', default=50, type=int,
                    help='Number of adversarial iterations')
parser.add_argument('--cuda', default=True,  dest='cuda',
                    action='store_true', help='Use cuda to train model')
parser.add_argument('--model_path', default='defense_models/deepspeech_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--labels_path', default='labels.json',
                    type=str, help='Labels path')
parser.add_argument('--eps', default=2000, type=float,
                    help='l infinity epsilon limit')
parser.add_argument('--adv_lr', default=0.1, type=float,
                    help='adversarial attack learning rate')
parser.add_argument('--manifest_filepath', type=str,
                    default='data/libri_test_clean_manifest.csv', help='The manifest filepath.')
if __name__ == '__main__':
    parser = add_decoder_args(parser)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    print("Loading checkpoint model %s" % args.model_path)
    package = torch.load(args.model_path,
                         map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model_package(package)
    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    labels_map = dict([(labels[i], i) for i in range(len(labels))])

    with open(args.manifest_filepath) as f:
        ids = f.readlines()
    ids = [x.strip().split(',') for x in ids]
    print(ids[49])
    #print(ids[6])
    #print(args.input)
    #print(len(args.input))
    fs, audio = wav.read(args.input)
    audio = torch.FloatTensor(audio).to(device)
    assert fs == 16000
    audio_length = len(audio)
    audio_length = torch.IntTensor([audio_length])
    
    decoder = BeamCTCDecoder(labels,lm_path = args.lm_path,alpha = args.alpha,beta = args.beta,cutoff_top_n = args.cutoff_top_n,cutoff_prob = args.cutoff_prob,beam_width = args.beam_width,num_processes = args.lm_workers)
    
    def transcripts(transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '').lower()
        return transcript

    def parse_transcript(transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '').lower()
        transcript = list(filter(None, [labels_map.get(x) for x in list(transcript)]))
        return transcript  # mapped in dictionary transcripts
    count =0
    for i,j in ids:
        if i == args.input:
            transcription = transcripts(j)
            print('The transcription is : {} '.format(transcription))
            targets = parse_transcript(j)
            targets = torch.IntTensor(targets)
            count +=1
        else:
            pass
    if count == 0:
        #print('audio file not found in the manifest choose a file from manifest')
        sys.exit('audio file not found in the manifest choose a file from manifest')

    adv_iterations_list = [40,50,60,70]
    attack = PGDAttack(model=model, audio_conf=audio_conf, device=device, iterations=args.adv_iterations,
                       optimizer_type='Adam', adv_lr=args.adv_lr, momentum=0.9)
    target_sizes = torch.IntTensor([len(targets)])
    model = model.to(device)
    model.train()
    ratios = [0.1, 0.2, 0.3, 0.5, 0.7,0.9]
    for ratio in ratios:
        audio = audio.reshape(1,-1)
        #print(audio.shape)
        #print(audio[0])
        #print(ratio*audio_length.item())
        audio_cut = audio[0][:int(ratio*audio_length.item())].reshape(1,-1)
        #print(audio_cut)
        #print(audio_cut.shape)
        

        spectrograms_cut = SpectrogramDataset(
            audio_conf=audio_conf, audios=audio_cut, normalize=True)
        spectrograms = SpectrogramDataset(
            audio_conf=audio_conf, audios=audio, normalize=True)
        #print(spectrograms_cut)
        #print(spectrograms)
        spectrogram_audio_cut = SpectrogramDataLoader(spectrograms_cut, 1)
        spectrogram_audio = SpectrogramDataLoader(spectrograms, 1)

        for _, (data) in enumerate(spectrogram_audio_cut):
                inputs_cut, input_percentages_cut = data
                input_sizes_cut = input_percentages_cut.mul_(
                    int(inputs_cut.size(3))).int()
        for _, (data) in enumerate(spectrogram_audio):
                inputs, input_percentages = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

        inputs_cut = inputs_cut.to(device)
        inputs = inputs.to(device)

        out_cut, output_sizes_cut = model(inputs_cut, input_sizes_cut)
        out, output_sizes = model(inputs, input_sizes)

        out_cut = F.softmax(out_cut, dim=-1)
        out = F.softmax(out, dim=-1)

        decoded_output_cut, _ = decoder.decode(out_cut, output_sizes_cut)
        # output shape should be (batch_x,seq_x,label_size)
        decoded_output, _ = decoder.decode(out, output_sizes)
        len_k = len(decoded_output_cut[0][0])
        #print(len_k)
        print('On natural dataset, the {k}th audiocut decoded output is : {out}'.format(
                k=ratio, out=decoded_output_cut[0][0]))
        print('On narural dataset,whole prediction part is : {}'.format(
                decoded_output[0][0][:len_k]))
    
      
    for i in adv_iterations_list:    
        #print(audio)
        #print(audio_length)
        #print(targets)
        #print(target_sizes)
        adv_audio = attack.attack(audios=audio, audio_lengths=audio_length, targets=targets,
                                          target_sizes=target_sizes, epsilon=args.eps, adv_iterations=i)
        adv_audio = adv_audio.detach()
        audio_length = torch.IntTensor([len(adv_audio[0])])
        for ratio in ratios:
            #print(int(ratio*audio_length))
            adv_audio_cut = adv_audio[0][:int(ratio*audio_length.item())].reshape(1,-1)
            #print(adv_audio_cut)
            spectrograms_cut = SpectrogramDataset(
                    audio_conf=audio_conf, audios=adv_audio_cut, normalize=True)
            spectrograms = SpectrogramDataset(
                    audio_conf=audio_conf, audios=adv_audio, normalize=True)
            adv_spectrogram_audio_cut = SpectrogramDataLoader(
                    spectrograms_cut, 1)
            adv_spectrogram_audio = SpectrogramDataLoader(spectrograms, 1)
            for _, (data1) in enumerate(adv_spectrogram_audio):
                inputs1, input_percentages1 = data1
                input_sizes1 = input_percentages1.mul_(int(inputs1.size(3))).int()
            for _, (data1) in enumerate(adv_spectrogram_audio_cut):
                inputs1_cut, input_percentages1_cut = data1
                input_sizes1_cut = input_percentages1_cut.mul_(int(inputs1_cut.size(3))).int()

            inputs1 = inputs1.to(device)
            inputs1_cut = inputs1_cut.to(device)
            #print(inputs.device)
            out1, output_sizes1 = model(inputs1, input_sizes1)
            out1_cut, output_sizes1_cut = model(inputs1_cut, input_sizes1_cut)
            out1 = F.softmax(out1, dim=-1)
            out1_cut = F.softmax(out1_cut, dim=-1)
            # output shape should be (batch_x,seq_x,label_size)
            decoded_output, _ = decoder.decode(out1, output_sizes1)
            decoded_output_cut, _ = decoder.decode(out1_cut, output_sizes1_cut)
            len_k = len(decoded_output_cut[0][0])
            print('for {i} number of adversarial iterations, the {k}th audiocut decoded output is : {out}'.format(
                    i=i, k=ratio, out=decoded_output_cut[0][0]))
            print('for {i} number of adversarial iterations, whole prediction part is: {out}'.format(
                    i=i, out=decoded_output[0][0][:len_k]))
    torch.cuda.empty_cache()
