import torch 
import argparse

from attack_madry import PGDAttack
from PGD_attack_dataloader import NaturalAudioDataLoader, AudiofileDataset, SpectrogramDataset,SpectrogramDataLoader
from model import DeepSpeech
import torch.nn.functional as F
from decoder import BeamCTCDecoder

from opts import add_decoder_args

parser = argparse.ArgumentParser(description='oneshot robustness test')
parser.add_argument('--in', type=str, dest="input",
                        required=True,
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")
parser.add_argument('--out',type=str,
                        required=True,
                        help="Path for the adversarial example(s)")
parser.add_argument('--adv_iterations', default=50, type=int, help='Number of adversarial iterations')
parser.add_argument('--cuda',default =True,  dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--model_path', default='defense_models/deepspeech_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--labels_path', default= 'labels.json', type=str, help='Labels path')
parser.add_argument('--eps', default=2000, type=float, help='l infinity epsilon limit')
parser.add_argument('--adv_lr', default=0.1, type=float, help='adversarial attack learning rate')
parser.add_argument('--manifest_filepath', type = str , default = 'libri_test_clean_manifest.csv',help ='The manifest filepath.')
if __name__ == '__main__':
    parser = add_decoder_args(parser)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    
    print("Loading checkpoint model %s" % args.model_path)
    package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model_package(package)
    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    labels_map = dict([(labels[i], i) for i in range(len(labels))])
    
    with open(args.manifest_filepath) as f:
        ids = f.readlines()
    ids = [x.strip().split(',') for x in ids]
    for i in range(len(args.input)):
        fs,audio = wav.read(args.input[i])
        audio = torch.FloatTensor(audio).to(device)
        assert fs == 16000
        audio_length = len(audio)

    def parse_transcript(transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '').lower()
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript #mapped in dictionary transcripts
    
    
    for i in ids:
        if ids[i][0] == args.input:
            transcription = transcripts(ids[i][1])
            print('The transcription is : {} '.format(transcription)) 
            targets = parse_transcript(ids[i][1])
            targets = torch.IntTensor(targets)
        else:
            pass
    
    adv_iterations_list = list(20,30,40,50)
    attack = PGDAttack(model = model,audio_conf = audio_conf ,device =  device , iterations = args.adv_iterations,optimizer_type = 'Adam',labels_path = args.labels_path,lr = args.adv_lr , momentum = 0.9)
    target_sizes = torch.IntTensor(len(targets))
    model = model.to(device)
    model.train()
    with torch.no_grad():
            spectrograms = SpectrogramDataset(audio_conf=audio_conf,audios=audio,normalize = True)
            spectrogram_audio = SpectrogramDataLoader(spectrograms,1)
            for _,(data) in enumerate(spectrogram_audio):
                inputs , input_percentages = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()


            # unflatten targets
            split_targets = []
            offset = 0
            
            inputs = inputs.to(device)

            out, output_sizes = model(inputs, input_sizes)

            out = F.softmax(out,dim=-1)
            decoded_output, _ = decoder.decode(out, output_sizes) #output shape should be (batch_x,seq_x,label_size)
            print('On natural dataset, the decoded output is : {out}'.format(decoded_output[0]))
            

            for i in adv_iterations_list:
            	adv_audio = attack.attack(audios = audio , audio_lengths = audio_length ,targets = targets , target_sizes = target_sizes, epsilon = args.eps, adv_iterations = i)
            	spectrograms = SpectrogramDataset(audio_conf=audio_conf,audios=adv_audio,normalize = True)
            	adv_spectrogram_audio = SpectrogramDataLoader(spectrograms,1)
            	for _,(data1) in enumerate(adv_spectrogram_audio):
                	inputs1 , input_percentages1 = data1
                	input_sizes1 = input_percentages1.mul_(int(inputs1.size(3))).int()
            	split_targets = []
            	offset = 0
            	
            	inputs = inputs.to(device)

            	out1, output_sizes1 = model(inputs1, input_sizes1)

            	out1 = F.softmax(out1,dim=-1)
            	decoded_output, _ = decoder.decode(out1, output_sizes1) #output shape should be (batch_x,seq_x,label_size)
            	print('for {i} number of adversarial iterations, the decoded output is : {out}'.format(i =i ,out = decoded_output[0]))
            torch.cuda.empty_cache()
    
