import argparse

import numpy as np
import torch
from tqdm import tqdm

from data.data_loader import SpectrogramDataset, AudioDataLoader
#from likelihood_decoder import BeamCTCDecoder
from model import DeepSpeech
from opts import add_decoder_args, add_inference_args

parser = argparse.ArgumentParser(description='Likelihood calculation')
parser = add_inference_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--model_path_benign', required =True,                                                                        
                            help='Location to save best validation model')
parser.add_argument('--model_path_adversarial', required =True,                                                                       
                            help='Location to save best validation model')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
no_decoder_args = parser.add_argument_group("No Decoder Options", "Configuration options for when no decoder is "
                                                                  "specified")
no_decoder_args.add_argument('--output-path', default=None, type=str, help="Where to save raw acoustic output")
parser = add_decoder_args(parser)
args = parser.parse_args()

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    print('Loading the benign model')
    model_1 = DeepSpeech.load_model(args.model_path_benign)
    model_1 = model_1.to(device)
    model_1.eval()

    print('Loading the adversarial model')
    model_2 = DeepSpeech.load_model(args.model_path_adversarial)  
    model_2 = model_2.to(device)                                                                                                              
    model_2.eval()

    labels = DeepSpeech.get_labels(model_1)
    audio_conf = DeepSpeech.get_audio_conf(model_1)

    if args.decoder == "beam":
        from likelihood_decoder import BeamCTCDecoder
        print('Beam decoder')
        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        print('Greedy decoder')
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    else:
         decoder = None
    #target_decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest, labels=labels,
                                      normalize=True)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    total_cer_1, total_wer_1, num_tokens_1, num_chars_1, total_cer_2, total_wer_2, num_tokens_2 ,num_chars_2 = 0, 0, 0, 0, 0, 0, 0, 0
    output_data_1 = []
    output_data_2 = []
    likelihood_correct, likelihood_wrong= [], []
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        inputs = inputs.to(device)
        out_1, output_sizes_1 = model_1(inputs, input_sizes)
        out_2, output_sizes_2 = model_2(inputs, input_sizes)
        if decoder is None:
            # add output to data array, and continue
            output_data_1.append((out_1.numpy(), output_sizes_1.numpy()))
            output_data_2.append((out_2.numpy(), output_sizes_2.numpy()))
            continue

        decoded_output_1, offset_1, scores_1, seq_len_1  = decoder.decode(out_1.data, output_sizes_1.data)
        decoded_output_2, offset_2, scores_2, seq_len_2  = decoder.decode(out_2.data, output_sizes_2.data)
        #print(scores_1)
        #print(decoded_output_1)
        
        target_strings = decoder.convert_to_transcripts(split_targets)
        #print(target_strings)
        for x in range(len(target_strings)):
            transcript_1, reference = decoded_output_1[x][0], target_strings[x][0]
            transcript_2 = decoded_output_2[x][0]
            wer_inst_1 = decoder.wer(transcript_1, reference)
            cer_inst_1 = decoder.cer(transcript_1, reference)
            total_wer_1 += wer_inst_1
            total_cer_1 += cer_inst_1
            num_tokens_1 += len(reference.split())
            num_chars_1 += len(reference)
            
            wer_inst_2 = decoder.wer(transcript_2, reference)                                                                                 
            cer_inst_2 = decoder.cer(transcript_2, reference)                                                                                 
            total_wer_2 += wer_inst_2
            total_cer_2 += cer_inst_2
            num_tokens_2 += len(reference.split()) 
            num_chars_2 += len(reference)

            if (transcript_1== reference) and (transcript_2 != reference):
                likelihood_wrong.extend([(scores_1[x][0]/num_chars_1).item()])
                
                
            elif (transcript_1== reference) and (transcript_2 == reference):
                likelihood_correct.extend([(scores_1[x][0]/num_chars_1).item()])
        
            else:
                pass
            

    if decoder is not None:
        wer_1 = float(total_wer_1) / num_tokens_1
        cer_1 = float(total_cer_1) / num_chars_1
        wer_2 = float(total_wer_2) / num_tokens_2
        cer_2 = float(total_cer_2) / num_chars_2

        print('The result for undefended model is\t'
              'Test Summary \t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(wer=wer_1 * 100, cer=cer_1 * 100))
        print('The result for defended model is\t'                                                                                          
              'Test Summary \t'                                                                                                               
              'Average WER {wer:.3f}\t'                                                                                                       
              'Average CER {cer:.3f}\t'.format(wer=wer_2 * 100, cer=cer_2 * 100))
        print(likelihood_correct)
        print(likelihood_wrong)
        np.save('likelihood/likelihood_correct.npy',likelihood_correct)
        np.save('likelihood/likelihood_wrong.npy',likelihood_wrong) 
    else:
        np.save(args.output_path, output_data)
