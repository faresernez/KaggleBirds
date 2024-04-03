import time
start = time.time()

import pandas as pd
import os
from pathlib import Path
import torchaudio
# import h5py
import torch
# torch.backends.cudnn.benchmark = True
import pandas as pd
import os
import numpy as np
from pathlib import Path
import torchaudio
# import h5py
import torch
# torch.backends.cudnn.benchmark = True
import shutil
import random
import torch.nn as nn
from torch import optim
from calibration import calibrate

softMax = nn.Softmax(dim=1)

class LSTM(nn.Module):
    """
    input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediciton_periods input to get_x_y_pairs
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)


        
        self.linear = nn.Linear(hidden_size, output_size)
        # self.linear2 = nn.Linear(output_size, 264)

        # self.soft = nn.Softmax(dim = 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, batch_size , hidden=None):
        if hidden==None:
            self.hidden = (torch.zeros(1,batch_size,self.hidden_size).to('cpu'),
                           torch.zeros(1,batch_size,self.hidden_size).to('cpu'))
        else:
            self.hidden = hidden
            
        """
        inputs need to be in the right shape as defined in documentation
        - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        
        lstm_out - will contain the hidden states from all times in the sequence
        self.hidden - will contain the current hidden state and cell state
        """
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # lstm_out, self.hidden = self.lstm(x.view(len(x),1,-1), 
        #                                   self.hidden)
        # print('ee')
        # print(lstm_out.shape)
        # print('lstm_out')
        # print(lstm_out.shape)
        predictions = self.linear(lstm_out)
        # print('pred shape')
        # print(predictions.shape)

        # predictions = self.relu(predictions)
        # print('pred ras nami')
        # print(predictions[:,-1,:].shape)

        # predictions = self.linear2(predictions)

        # predictions = self.soft(predictions)

        # preds = self.soft(predictions[:,-1,:])
        # print(predictions4[:,-1,:])
        
        return predictions[:,-1,:], self.hidden

processing_done = True

if not processing_done:
    os.makedirs('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/infer/hidden64000/')
    os.makedirs('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/infer/subtracks64000/')

if not processing_done:

    ### WAV2VEC MODEL ###

    model_checkpoint =   'facebook/wav2vec2-base-960h' 
    from datasets import load_dataset, load_metric, Audio

    common_voice_train = load_dataset("common_voice", "tr", split="train+validation")
    common_voice_test = load_dataset("common_voice", "tr", split="test")

    common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

    import re
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

    def remove_special_characters(batch):
        batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
        return batch

    def extract_all_chars(batch):
        all_text = " ".join(batch["sentence"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocab_train = common_voice_train.map(
    extract_all_chars, batched=True, 
    batch_size=-1, keep_in_memory=True, 
    remove_columns=common_voice_train.column_names
    )
    vocab_test = common_voice_test.map(
    extract_all_chars, batched=True, 
    batch_size=-1, keep_in_memory=True, 
    remove_columns=common_voice_test.column_names
    )

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    len(vocab_dict)

    import json
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_checkpoint)

    tokenizer_type = config.model_type if config.tokenizer_class is None else None
    config = config if config.tokenizer_class is not None else None

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
    "./",
    config=config,
    tokenizer_type=tokenizer_type,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
    )

    del vocab_list
    del vocab_train
    del vocab_test
    del common_voice_train
    del common_voice_test

    from transformers import AutoFeatureExtractor, AutoModelForCTC , Wav2Vec2Model
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    from transformers import Wav2Vec2Processor


    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    del feature_extractor
    del tokenizer

    model = Wav2Vec2Model.from_pretrained(model_checkpoint).to('cpu')

    def wav2vec_processing():
        path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/infer/subtracks64000/'
        saving_path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/infer/hidden64000/'

        batch_size = 64
        # # n_subtracks = 8762
        # n_loops = n_subtracks // batch_size
        # reste = n_subtracks % batch_size

        # mode = 'train/'
        # path += mode
        l = os.listdir(path)
        # saving_path += mode
        n_subtracks = len(l)
        n_loops = n_subtracks // batch_size
        reste = n_subtracks % batch_size
        # i = 0

        # print('model device')
        # print(model.device)

        for k in range(n_loops):
            t = torch.load(path + l[0 + batch_size*k], map_location='cpu')
            for i in range(batch_size - 1):
                x = torch.load(path + l[i + 1 + batch_size*k], map_location='cpu')
                t = torch.cat((t,x),dim=0)
            # print('tensor device')
            # print(t.device)
            # # print(t.shape)
            # end1 = time.time()
            # print('loading')
            # print(end1 - start)

            # start = time.time()
            with torch.no_grad():
                input_values = processor(t, return_tensors="pt",sampling_rate = 16000).input_values.to('cpu')  # Batch size 1
                # print(input_values.shape)
                # print('input values')
                # print(input_values.device)

                # print('ok')
                # print(input_values.squeeze(0).shape)
                hidden_states = model(input_values.squeeze(0)).last_hidden_state
                # print(hidden_states.shape)
            for t in range(batch_size):
                torch.save(hidden_states[t], saving_path + l[t + batch_size*k])
        if reste != 0:
            done = batch_size * n_loops
            t = torch.load(path + l[done], map_location='cpu')
            for i in range(reste - 1):
                x = torch.load(path + l[i + 1 + done], map_location='cpu')
                t = torch.cat((t,x),dim=0)
            # print(t.shape)

            with torch.no_grad():
                input_values = processor(t, return_tensors="pt",sampling_rate = 16000).input_values.to('cpu')  # Batch size 1
                # print(input_values.shape)

                # print('ok')
                # print(input_values.squeeze(0).shape)
                hidden_states = model(input_values.squeeze(0)).last_hidden_state
                # print(hidden_states.shape)

            for t in range(reste):
                torch.save(hidden_states[t], saving_path + l[t + done])
            # torch.save(hidden_states, saving_path + l[done])
        # shutil.rmtree(path)
        # os.makedirs(path)

### END WAV2VEC MODEL ###

### general

general_species_dict = {}
size_dict = {}
pathlist = Path(r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\train_audio').glob('*')
i = 0
for path in pathlist:
    path_in_str = str(path)
    species = path_in_str[60:]
    general_species_dict[species] = i
    s = 0
    for ele in os.scandir(path_in_str):
        s += os.path.getsize(ele)
    size_dict[species] = s
    i += 1
sorted_size_dict = sorted(size_dict.items(), key=lambda x:x[1], reverse=True)
converted_dict = dict(sorted_size_dict)

sorted_list = list(converted_dict.keys())

general_species_dict_inverted = {}
for key,value in general_species_dict.items():
#     print(ke)
    general_species_dict_inverted[value] = key

batch0 = {}
batch1 = {}
for j in range(11):
    batch1['list_' + str(2*j)] = [sorted_list[12*2*j] ,sorted_list[12*2*j + 2] ,sorted_list[12*2*j + 4] ,sorted_list[12*2*j +6] ,
                                sorted_list[12*2*j + 8] ,sorted_list[12*2*j + 10] ,sorted_list[12*2*j + 12] ,sorted_list[12*2*j + 14] ,
                                sorted_list[12*2*j + 16] ,sorted_list[12*2*j + 18] ,sorted_list[12*2*j + 20] ,sorted_list[12*2*j + 22]  ]
    
    batch1['list_' + str(2*j + 1)] =  [sorted_list[12*2*j + 1] ,sorted_list[12*2*j + 3] ,sorted_list[12*j + 5] ,sorted_list[12*2*j +7] ,
                                sorted_list[12*2*j + 9] ,sorted_list[12*2*j + 11] ,sorted_list[12*2*j + 13] ,sorted_list[12*2*j + 15] ,
                                sorted_list[12*2*j + 17] ,sorted_list[12*2*j + 19] ,sorted_list[12*2*j + 21] ,sorted_list[12*2*j + 23]  ]
    
    batch0['list_' + str(2*j)] = [sorted_list[12*j] ,sorted_list[12*j + 1] ,sorted_list[12*j + 2] ,sorted_list[12*j +3] ,
                                sorted_list[12*j + 4] ,sorted_list[12*j + 5] ,sorted_list[12*j + 6] ,sorted_list[12*j + 7] ,
                                sorted_list[12*j + 8] ,sorted_list[12*j + 9] ,sorted_list[12*j + 10] ,sorted_list[12*j + 11]  ]
    
    batch0['list_' + str(2*j + 1)] = [sorted_list[12*(j+1)] ,sorted_list[12*(j+1) + 1] ,sorted_list[12*(j+1) + 2] ,sorted_list[12*(j+1) +3] ,
                                sorted_list[12*(j+1) + 4] ,sorted_list[12*(j+1) + 5] ,sorted_list[12*(j+1) + 6] ,sorted_list[12*(j+1) + 7] ,
                                sorted_list[12*(j+1) + 8] ,sorted_list[12*(j+1) + 9] ,sorted_list[12*(j+1) + 10] ,sorted_list[12*(j+1) + 11]  ]
###end gerneral

### preprocess 

if not processing_done:
    steps_per_subtrack = 32000
    flush_every = 150
    limit = 14000

    transform = torchaudio.transforms.Resample(32000, 16000)
    file = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/test_soundscapes/soundscape_29201.ogg'

    data, samplerate = torchaudio.load(file, normalize=True)
    # print(samplerate)
    data = transform(data)
    track_length = data.shape[1]
    for i in range(track_length // steps_per_subtrack):
        # with torch.no_grad():
            # input_values = processor(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], return_tensors="pt",sampling_rate = 16000).input_values.to('cuda:0')  # Batch size 1
            # hidden_states = model(input_values.squeeze(0)).last_hidden_state
            # subtrack_index = str(k)
            # torch.save(hidden_states, 'data/subtracks64000/train/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
            # k += 1
        subtrack_index_2i = str(2*i)
        subtrack_index_2iplus1 = str(2*i + 1)
        torch.save(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], 'data/infer/subtracks64000/' + subtrack_index_2i + '.pt')
        if i != track_length // steps_per_subtrack - 1 :
            torch.save(data[:,int(steps_per_subtrack * (i + 0.5)):int(steps_per_subtrack * (i + 1.5))], 'data/infer/subtracks64000/' + subtrack_index_2iplus1 + '.pt')
#             torch.save(data[:,i*int(steps_per_subtrack/2):(i+1)*int(steps_per_subtrack/2)], '/kaggle/tmp/subtracks64000/' + subtrack_index_2iplus1 + '.pt')
    # print(i)
    wav2vec_processing()

# if not processing_done:
#     steps_per_subtrack = 32000
#     flush_every = 150
#     limit = 14000

#     transform = torchaudio.transforms.Resample(32000, 16000)
#     file = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/test_soundscapes/soundscape_29201.ogg'

#     data, samplerate = torchaudio.load(file, normalize=True)
#     # print(samplerate)
#     data = transform(data)
#     track_length = data.shape[1]
#     for i in range(track_length // steps_per_subtrack):
#         # with torch.no_grad():
#             # input_values = processor(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], return_tensors="pt",sampling_rate = 16000).input_values.to('cuda:0')  # Batch size 1
#             # hidden_states = model(input_values.squeeze(0)).last_hidden_state
#             # subtrack_index = str(k)
#             # torch.save(hidden_states, 'data/subtracks64000/train/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
#             # k += 1
#         subtrack_index = str(i)
#         torch.save(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], 'data/infer/subtracks64000/' + subtrack_index + '.pt')
#     # print(i)
#     wav2vec_processing()

    del model
    del processor
    del data

### end preprocess

### inference

predictions = []

for i in range(2):
    d = {}
    for key in list(general_species_dict.keys()):
        d[key] =  []
    predictions.append(d)

# print('predictions')
# print(predictions)
# print(predictions[0])

# q_hat = []
q_hat = 0.9

path_test = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/infer/hidden64000/'
l_test = os.listdir(path_test)


batch_size = 256
nelem = len(l_test)
print(nelem)
n_loops = nelem // batch_size
reste = nelem % batch_size

# quelbatch = 0
# n_list = 0

for quelbatch in range(2):
    for n_list in range(22):
        start1 = time.time()
        print(quelbatch, '  ', n_list)
        model_path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/'
        if quelbatch == 0:
            model_path += 'batch1/nlist_'
        else:
            model_path += 'batch2/nlist_'
            
        model_path += str(n_list)

        model = LSTM(input_size=768, hidden_size=256, output_size=12)
#         model = LSTM(input_size=768, hidden_size=256, output_size=12).cuda()
        model.load_state_dict(torch.load(model_path))
        model.eval()
#         model.cuda()
        # print(model.device)

        with torch.no_grad():
            for j in range(n_loops):
                t = torch.load(path_test + str(j*batch_size) + '.pt' , map_location='cpu').unsqueeze(dim=0)
#                 t = torch.load(path_test + str(j*batch_size) + '.pt').unsqueeze(dim=0).cuda()
                # y = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_test[j*batch_size],species_dict)
                for i in range(batch_size - 1):
                        x = torch.load(path_test + str(j*batch_size + i + 1) + '.pt' , map_location='cpu').unsqueeze(dim=0)
#                         x = torch.load(path_test + str(j*batch_size + i + 1) + '.pt').unsqueeze(dim=0).cuda()
                        # z = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_test[j*batch_size + i + 1],species_dict)
                        t = torch.cat((t,x),dim=0)
                        # print('b')
                        # print(t.shape)
                        # y = torch.cat((y,z),dim=0)
#                 print('a')
#                 print(t.device)
#                 print(model.device)
                y_hat, _ = model(t, batch_size, None)
#                 print('y')
#                 print(y_hat.device)
                if j == 0:
                    ypred_test = softMax(y_hat)
                    print(ypred_test.device)
                    # yt_test = y
                else:
                    ypred_test = torch.cat((ypred_test,softMax(y_hat)),dim=0)
                    print(ypred_test.device)
                    # yt_test = torch.cat((yt_test,y),dim=0)
            if reste != 0:
                done = batch_size * n_loops
                t = torch.load(path_test + str(done) + '.pt' , map_location='cpu').unsqueeze(dim=0)
#                 t = torch.load(path_test + str(done) + '.pt').unsqueeze(dim=0).cuda()
                # y = torch.as_tensor(pathToSpecies(l_train[done],species_dict)).to('cpu')
                # y = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_test[done],species_dict)
                for i in range(reste - 1):
                    x = torch.load(path_test + str(done + i + 1) + '.pt' , map_location='cpu').unsqueeze(dim=0)
#                     x = torch.load(path_test + str(done + i + 1) + '.pt' ).unsqueeze(dim=0).cuda()
                    # z = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_test[done + i + 1],species_dict)
                    t = torch.cat((t,x),dim=0)
                    # y = torch.cat((y,z),dim=0)
#                 print('b')
#                 print(t.device)
#                 print(model.device)
                y_hat, _ = model(t, reste, None)
                
                ypred_test = torch.cat((ypred_test,softMax(y_hat)),dim=0)
                print(ypred_test.device)
                # yt_test = torch.cat((yt_test,y),dim=0).cpu().numpy()
                ypred_test.cpu().numpy()
        # print(ypred_test.shape)

        all_pred = []
        for i,elem in enumerate(ypred_test):
            pred = []
            # print(elem)
            for classe,soft in enumerate(elem):
                if (1 - soft) <= q_hat:
                    pred.append(classe)
            # print(pred)
            all_pred.append(pred)

        if quelbatch  == 0:
            batch = batch0
        else: 
            batch = batch1

        training_list = batch['list_' + str(n_list)]
        # print(training_list)
        # species_dict = {}

        for l in all_pred:
            for classe in range(12):
                if classe in l:
                    predictions[quelbatch][training_list[classe]].append(1.)
                else:
                    predictions[quelbatch][training_list[classe]].append(0.)
        # print(predictions[quelbatch])
        end1 = time.time()
        print(end1 - start1)

final_prediction = np.zeros(264)

for i in range(264):
    for j in range(6):
#         print(predictions[0][general_species_dict_inverted[i]])
#         print(predictions[1][general_species_dict_inverted[i]])
        if predictions[0][general_species_dict_inverted[i]] != [] and predictions[1][general_species_dict_inverted[i]] != []:
#             print('ok')
            if  predictions[0][general_species_dict_inverted[i]][j+10] == predictions[1][general_species_dict_inverted[i]][j+10] and predictions[0][general_species_dict_inverted[i]][j] != 0:
                final_prediction[i] += 1
print(final_prediction)

prediction_dic = {}
for i in range(120):  
#     print(i)
    prediction_dic[i] = []
    final_prediction = np.zeros(264)
    if i == 0:
        for k in range(264):
            for j in range(5):
                if predictions[0][general_species_dict_inverted[k]] != [] and predictions[1][general_species_dict_inverted[k]] != []:
#                     print(len(predictions[1][general_species_dict_inverted[k]]))
                    if  predictions[0][general_species_dict_inverted[k]][5*i+j] == predictions[1][general_species_dict_inverted[k]][5*i+j] and predictions[0][general_species_dict_inverted[k]][5*i+j] != 0:
                        final_prediction[k] += 1
        for ind,elem in enumerate(final_prediction):
            if elem > 5:
                prediction_dic[i].append(general_species_dict_inverted[ind])
    elif i < 119:
        for k in range(264):
#             print(k)
            for j in range(6):
                if predictions[0][general_species_dict_inverted[k]] != [] and predictions[1][general_species_dict_inverted[k]] != []:
                    if  predictions[0][general_species_dict_inverted[k]][5*i+j-1] == predictions[1][general_species_dict_inverted[k]][5*i+j-1] and predictions[0][general_species_dict_inverted[k]][5*i+j-1] != 0:
                        final_prediction[k] += 1
        for ind,elem in enumerate(final_prediction):
            if elem > 4:
                prediction_dic[i].append(general_species_dict_inverted[ind])
    else:
        for k in range(264):
#             print(k)
            for j in range(5):
                if predictions[0][general_species_dict_inverted[k]] != [] and predictions[1][general_species_dict_inverted[k]] != []:
                    if  predictions[0][general_species_dict_inverted[k]][5*i+j-1] == predictions[1][general_species_dict_inverted[k]][5*i+j-1] and predictions[0][general_species_dict_inverted[k]][5*i+j-1] != 0:
                        final_prediction[k] += 1
        for ind,elem in enumerate(final_prediction):
            if elem > 4:
                prediction_dic[i].append(general_species_dict_inverted[ind])
# print(prediction_dic)

id = 'soundscape_29201_'
def newrow(i):
    id = 'soundscape_29201_'
    d = {'row_id' : id + str(5*(i+1))}
    for key in list(general_species_dict.keys()):
        d[key] = 0.
    return d

sample_sub = pd.read_csv("C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/sample_submission.csv")
sample_sub
for i in range(120):
    if i < 3:
        if prediction_dic[i] != []:
            for species in prediction_dic[i]:
                sample_sub.at[i, species] = 1
    else:
        dic = newrow(i)
#         print(dic)
        # sample_sub = sample_sub.append(d, ignore_index=True)
        sample_sub = pd.concat([sample_sub, pd.DataFrame([dic])], ignore_index=True)
        if prediction_dic[i] != []:
            for species in prediction_dic[i]:
                sample_sub.at[i, species] = 1
        sample_sub.at[i, 'row_id'] = id + str(5*(i+1))
    # print(sample_sub[i])
        # sample_sub = pd.concat([sample_sub, pd.DataFrame([d])], ignore_index=True)
# print(sample_sub)
sample_sub.to_csv('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/results.csv',index=False)

end = time.time()
print('finished in ',end - start)




### inference

# predictions = []

# for i in range(2):
#     d = {}
#     for key in list(general_species_dict.keys()):
#         d[key] =  []
#     predictions.append(d)

# # q_hat = []
# q_hat = 0.75

# path_test = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/infer/hidden64000/'
# l_test = os.listdir(path_test)

# batch_size = 64
# nelem = len(l_test)
# print(nelem)
# n_loops = nelem // batch_size
# reste = nelem % batch_size

# quelbatch = 0
# n_list = 0

# # for quelbatch in range(2):
# #     for n_list in range(22):

# model = LSTM(input_size=768, hidden_size=256, output_size=12).to('cpu')
# model.load_state_dict(torch.load(r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\models\batch1\nlist_0'))
# model.eval()

# with torch.no_grad():
#     for j in range(n_loops):
#         t = torch.load(path_test + l_test[j*batch_size], map_location='cpu').unsqueeze(dim=0)
#         # y = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_test[j*batch_size],species_dict)
#         for i in range(batch_size - 1):
#                 x = torch.load(path_test + l_test[j*batch_size + i + 1], map_location='cpu').unsqueeze(dim=0)
#                 # z = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_test[j*batch_size + i + 1],species_dict)
#                 t = torch.cat((t,x),dim=0)
#                 # print('b')
#                 # print(t.shape)
#                 # y = torch.cat((y,z),dim=0)
#         y_hat, _ = model(t, batch_size, None)
#         if j == 0:
#             ypred_test = y_hat
#             # yt_test = y
#         else:
#             ypred_test = torch.cat((ypred_test,y_hat),dim=0)
#             # yt_test = torch.cat((yt_test,y),dim=0)
#     if reste != 0:
#         done = batch_size * n_loops
#         t = torch.load(path_test + l_test[done], map_location='cpu').unsqueeze(dim=0)
#         # y = torch.as_tensor(pathToSpecies(l_train[done],species_dict)).to('cpu')
#         # y = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_test[done],species_dict)
#         for i in range(reste - 1):
#             x = torch.load(path_test + l_test[done + i + 1], map_location='cpu').unsqueeze(dim=0)
#             # z = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_test[done + i + 1],species_dict)
#             t = torch.cat((t,x),dim=0)
#             # y = torch.cat((y,z),dim=0)
#         y_hat, _ = model(t, reste, None)
#         ypred_test = torch.cat((ypred_test,y_hat),dim=0)
#         # yt_test = torch.cat((yt_test,y),dim=0).cpu().numpy()
#         ypred_test.cpu().numpy()
# # print(ypred_test.shape)

# all_pred = []
# for i,elem in enumerate(ypred_test):
#     pred = []
#     # print(elem)
#     for classe,soft in enumerate(elem):
#         if (1 - soft) <= q_hat:
#             pred.append(classe)
#     # print(pred)
#     all_pred.append(pred)

# if quelbatch  == 0:
#     batch = batch0
# else: 
#     batch = batch1
        
# training_list = batch['list_' + str(n_list)]
# # print(training_list)
# # species_dict = {}

# for l in all_pred:
#     for classe in range(12):
#         if classe in l:
#             predictions[quelbatch][training_list[classe]].append(1.)
#         else:
#             predictions[quelbatch][training_list[classe]].append(0.)
# # print(predictions[quelbatch])

# general_species_dict_inverted = {}
# for key,value in general_species_dict_inverted.items():
#     general_species_dict_inverted[value] = key

# final_prediction = np.zeros(264)

# for i in range(264):
#     for j in range(nelem):
#         if  predictions[0][general_species_dict_inverted[i]][j] == predictions[1][general_species_dict_inverted[i]][j] and predictions[0][general_species_dict_inverted[i]][j] != 0:
#             final_prediction[i] += 1






