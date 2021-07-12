import sys
import torch
import argparse
from tqdm import tqdm
import scipy.io.wavfile as wav

from attack_madry_signed import PGDAttack
from PGD_attack_dataloader import NaturalAudioDataLoader, AudiofileDataset, SpectrogramDataset, SpectrogramDataLoader, NaturalAudioDataLoader_cut, AudiofileDataset_cut
from model import DeepSpeech
import torch.nn.functional as F
from decoder import BeamCTCDecoder

from opts import add_decoder_args
parser = argparse.ArgumentParser(description='temporal modelling check')

parser.add_argument('--batch-size', default=16, type=int, help='Batch size for training')

parser.add_argument('--num-workers', default=1, type=int, help='Number of workers used in data-loading')

parser.add_argument('--adv_iterations', default=50, type=int,
                    help='Number of adversarial iterations')
parser.add_argument('--cuda',dest='cuda',
                    action='store_true', help='Use cuda to train model')
parser.add_argument('--model_path', default='defense_models/deepspeech_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--labels_path', default='labels.json',
                    type=str, help='Labels path')
parser.add_argument('--eps', default=200, type=float,
                    help='l infinity epsilon limit')
parser.add_argument('--adv_lr', default=0.5, type=float,
                    help='adversarial attack learning rate')
parser.add_argument('--manifest_filepath', type=str,
                    default='data/temporal_dependency.csv', help='The manifest filepath.')
if __name__ == '__main__':
    parser = add_decoder_args(parser)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    print("Loading checkpoint model %s" % args.model_path)
    package = torch.load(args.model_path,
                         map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model_package(package)
    labels = DeepSpeech.get_labels(model)
    
    model = model.to(device)
    
    model.train()
    
    audio_conf = DeepSpeech.get_audio_conf(model)
    
    audio_dataset = AudiofileDataset(audio_conf=audio_conf, manifest_filepath=args.manifest_filepath, labels=labels,
                                                  normalize=True, augment=False)
    audio_loader =  NaturalAudioDataLoader(audio_dataset, batch_size=args.batch_size,num_workers=args.num_workers)

    decoder = BeamCTCDecoder(labels,lm_path = args.lm_path,alpha = args.alpha,beta = args.beta,cutoff_top_n = args.cutoff_top_n,cutoff_prob = args.cutoff_prob,beam_width = args.beam_width,num_processes = args.lm_workers)
    attack = PGDAttack(model=model, audio_conf=audio_conf, device=device, iterations=args.adv_iterations,
                                   optimizer_type='dam', adv_lr=args.adv_lr, momentum=0.9)
    adv_iterations_list = [60,70]
    #ratios = [0.1, 0.2, 0.3, 0.5, 0.7,0.9]
    ratios = [1./2,2./3,3./4]
    total_cer_natural , total_wer_natural = 0, 0
    avg_cer_natural, avg_wer_natural,avg_cer_adv, avg_wer_adv = 0, 0, 0, 0
    d,adv_d = {},{}
    for i, (data) in tqdm(enumerate(audio_loader), total=len(audio_loader)):
        audios , audio_lengths, targets , target_sizes = data
        audios = audios.to(device)
        spectrograms = SpectrogramDataset(audio_conf=audio_conf, audios=audios, normalize=True)
        spectrogram_audio = SpectrogramDataLoader(spectrograms, args.batch_size)
    
        for _,(data1) in enumerate(spectrogram_audio):
            inputs , input_percentages = data1
            input_sizes = input_percentages.mul_(int(inputs.size(3))).short()
            
        inputs = inputs.to(device)
        out, output_sizes = model(inputs, input_sizes)
        out = F.softmax(out, dim=-1)
        decoded_output, _ = decoder.decode(out, output_sizes)
        for ratio in ratios:
            audio_cut_dataset = AudiofileDataset_cut(audios = audios, ratio = ratio)
            audio_cut_dataloader = NaturalAudioDataLoader_cut(audio_cut_dataset, batch_size = args.batch_size,num_workers = args.num_workers)
            for j,(data_cut) in enumerate(audio_cut_dataloader):
                audio_cut , audio_cut_lengths = data_cut
                spectrograms_cut = SpectrogramDataset(audio_conf=audio_conf, audios=audio_cut, normalize=True)
                spectrogram_audio_cut = SpectrogramDataLoader(spectrograms_cut, args.batch_size)
                
                for _,(data2) in enumerate(spectrogram_audio_cut):
                    inputs_cut , input_percentages_cut = data2
                    input_sizes_cut = input_percentages_cut.mul_(int(inputs_cut.size(3))).short()

            inputs_cut = inputs_cut.to(device)
            out_cut, output_sizes_cut = model(inputs_cut, input_sizes_cut)
            out_cut = F.softmax(out_cut, dim=-1)
            decoded_output_cut, _ = decoder.decode(out_cut, output_sizes_cut)
            wer , cer =0 ,0
            for x in range(len(decoded_output)):
                len_k = len(decoded_output_cut[x][0])
                transcript, transcript_cut  = decoded_output[x][0][:len_k], decoded_output_cut[x][0]
                if not len(transcript)== 0:
                    wer += decoder.wer(transcript, transcript_cut) / float(len(transcript.split()))
                    cer += decoder.cer(transcript, transcript_cut) / float(len(transcript))
                    
                elif len(transcript) == 0 and len(transcript_cut)==0:
                    pass
            try:
                d[(ratio,'wer')] += wer
                d[(ratio,'cer')] += cer
            except:
                d[(ratio,'cer')] = cer
                d[(ratio,'wer')] = wer
            #ratio_cer += cer/len(audio_cut_dataloader.dataset)
            #ratio_wer += wer/len(audio_cut_dataloader.dataset)
            total_cer_natural += cer/len(ratios)
            total_wer_natural += wer/len(ratios)
        
        
        avg_adv_iter_cer ,avg_adv_iter_wer = 0, 0
        for i in adv_iterations_list:    
            adv_audio = attack.attack(audios=audios, audio_lengths=audio_lengths, targets=targets,
                                          target_sizes=target_sizes, epsilon=args.eps, adv_iterations=i)
            adv_audio = adv_audio.detach()
            #print(adv_audio.device)
            spectrograms_adv = SpectrogramDataset(audio_conf=audio_conf, audios=adv_audio, normalize=True)
            spectrogram_audio_adv = SpectrogramDataLoader(spectrograms, batch_size = args.batch_size)
            for _,(data_adv) in enumerate(spectrogram_audio):
                inputs_adv , input_percentages_adv = data_adv
                input_sizes_adv = input_percentages_adv.mul_(int(inputs_adv.size(3))).short()
            inputs_adv = inputs_adv.to(device)
            out1, output_sizes1 = model(inputs_adv, input_sizes_adv)
            out1 = F.softmax(out1, dim=-1)
            decoded_output, _ = decoder.decode(out1, output_sizes1)
            ratio_cer_adv, ratio_wer_adv=0 ,0
            for ratio in ratios:
                #print(len(ratios),ratio)
                adv_audio_cut_dataset = AudiofileDataset_cut(audios = adv_audio, ratio = ratio)
                adv_audio_cut_dataloader = NaturalAudioDataLoader_cut(adv_audio_cut_dataset, batch_size = args.batch_size,num_workers = args.num_workers)
                for j,(adv_data_cut) in enumerate(adv_audio_cut_dataloader):
                    #print(len(adv_audio_cut_dataloader.dataset))
                    adv_audio_cut , adv_audio_cut_lengths = adv_data_cut
                    spectrograms_adv_cut = SpectrogramDataset(audio_conf=audio_conf, audios=adv_audio_cut, normalize=True)
                    spectrogram_audio_adv_cut = SpectrogramDataLoader(spectrograms_adv_cut, batch_size =args.batch_size)
                    for _,(data2_adv) in enumerate(spectrogram_audio_adv_cut):
                        inputs_adv_cut , input_percentages_adv_cut = data2_adv
                        input_sizes_adv_cut = input_percentages_adv_cut.mul_(int(inputs_adv_cut.size(3))).short()

                inputs_adv = inputs_adv.to(device)
                inputs_adv_cut = inputs_adv_cut.to(device)
                out1_cut, output_sizes1_cut = model(inputs_adv_cut, input_sizes_adv_cut)
                out1_cut = F.softmax(out1_cut, dim=-1)
                decoded_output_cut, _ = decoder.decode(out1_cut, output_sizes1_cut)
                cer_adv , wer_adv =0 ,0
                for x in range(len(decoded_output)):
                    len_k = len(decoded_output_cut[x][0])
                    transcript, transcript_cut  = decoded_output[x][0][:len_k], decoded_output_cut[x][0]
                    if not len(transcript)== 0:
                        wer_adv += decoder.wer(transcript, transcript_cut) / float(len(transcript.split()))
                        cer_adv += decoder.cer(transcript, transcript_cut) / float(len(transcript))
                        #print(transcript ,decoder.cer(transcript, transcript_cut) / float(len(transcript)))
                        #print(transcript_cut)
                    elif (len(transcript) == 0 and len(transcript_cut) ==0):
                        pass
                try:
                    adv_d[(ratio,'wer')] += wer_adv/float(len(adv_iterations_list))
                    adv_d[(ratio,'cer')] += cer_adv/float(len(adv_iterations_list))
                except:
                    adv_d[(ratio,'cer')] = cer_adv/float(len(adv_iterations_list))
                    adv_d[(ratio,'wer')] = wer_adv/float(len(adv_iterations_list))
                ratio_cer_adv += cer_adv/float(len(ratios))
                ratio_wer_adv += wer_adv/float(len(ratios))
                #print('Ratio',ratio_cer_adv,ratio_wer_adv)
            avg_adv_iter_cer += ratio_cer_adv/float(len(adv_iterations_list))
            avg_adv_iter_wer += ratio_wer_adv/float(len(adv_iterations_list))
            #print('iter_avg_************',avg_adv_iter_cer, avg_adv_iter_wer)

        avg_cer_natural += total_cer_natural/float(len(audio_loader.dataset))
        avg_wer_natural += total_wer_natural/float(len(audio_loader.dataset))
        avg_cer_adv +=  avg_adv_iter_cer/float(len(audio_loader.dataset))
        avg_wer_adv += avg_adv_iter_wer/float(len(audio_loader.dataset))
    print('the average cer and wer on benign data are : {cer} and {wer}'.format(cer = avg_cer_natural, wer = avg_wer_natural))
    print('the average cer and wer on adversarial data iterations are : {cer} and {wer}'.format(n = i, cer = avg_cer_adv , wer = avg_wer_adv))
    print('benign')
    print([(k,d[k]/float(len(audio_loader.dataset))) for k in d.keys()])
    print('*'*100)
    print('adversarial')
    print([(k,adv_d[k]/float(len(audio_loader.dataset))) for k in adv_d.keys()])
    torch.cuda.empty_cache()
