import pandas as pd
import os
from pathlib import Path
import torchaudio
# import h5py
import torch
torch.backends.cudnn.benchmark = True
import pandas as pd
import os
import time
import numpy as np
from pathlib import Path
import torchaudio
# import h5py
import torch
torch.backends.cudnn.benchmark = True
import shutil
import random
import torch.nn as nn
from torch import optim
from calibration import calibrate


### WAV2VEC MODEL ###

# model_checkpoint =   'facebook/wav2vec2-base-960h' 
# from datasets import load_dataset, load_metric, Audio

# common_voice_train = load_dataset("common_voice", "tr", split="train+validation")
# common_voice_test = load_dataset("common_voice", "tr", split="test")

# common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
# common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

# import re
# chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

# def remove_special_characters(batch):
#     batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
#     return batch

# def extract_all_chars(batch):
#   all_text = " ".join(batch["sentence"])
#   vocab = list(set(all_text))
#   return {"vocab": [vocab], "all_text": [all_text]}

# vocab_train = common_voice_train.map(
#   extract_all_chars, batched=True, 
#   batch_size=-1, keep_in_memory=True, 
#   remove_columns=common_voice_train.column_names
# )
# vocab_test = common_voice_test.map(
#   extract_all_chars, batched=True, 
#   batch_size=-1, keep_in_memory=True, 
#   remove_columns=common_voice_test.column_names
# )

# vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

# vocab_dict = {v: k for k, v in enumerate(vocab_list)}
# vocab_dict

# vocab_dict["|"] = vocab_dict[" "]
# del vocab_dict[" "]

# vocab_dict["[UNK]"] = len(vocab_dict)
# vocab_dict["[PAD]"] = len(vocab_dict)
# len(vocab_dict)

# import json
# with open('vocab.json', 'w') as vocab_file:
#     json.dump(vocab_dict, vocab_file)

# from transformers import AutoConfig

# config = AutoConfig.from_pretrained(model_checkpoint)

# tokenizer_type = config.model_type if config.tokenizer_class is None else None
# config = config if config.tokenizer_class is not None else None

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(
#   "./",
#   config=config,
#   tokenizer_type=tokenizer_type,
#   unk_token="[UNK]",
#   pad_token="[PAD]",
#   word_delimiter_token="|",
# )

# del vocab_list
# del vocab_train
# del vocab_test
# del common_voice_train
# del common_voice_test

# from transformers import AutoFeatureExtractor, AutoModelForCTC , Wav2Vec2Model
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

# from transformers import Wav2Vec2Processor


# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# del feature_extractor
# del tokenizer

# model = Wav2Vec2Model.from_pretrained(model_checkpoint).to('cuda:0')

from transformers import AutoConfig , Wav2Vec2Processor , Wav2Vec2FeatureExtractor , Wav2Vec2Model

config = AutoConfig.from_pretrained('C:/Users/fares/OneDrive/Bureau/kaggleBirds/wav2vec/')

tokenizer_type = config.model_type if config.tokenizer_class is None else None
config = config if config.tokenizer_class is not None else None

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
  'C:/Users/fares/OneDrive/Bureau/kaggleBirds/wav2vec/',
  config=config,
  tokenizer_type=tokenizer_type,
  unk_token="[UNK]",
  pad_token="[PAD]",
  word_delimiter_token="|",
)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("C:/Users/fares/OneDrive/Bureau/kaggleBirds/wav2vec/preprocessor_config.json")

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = Wav2Vec2Model.from_pretrained('C:/Users/fares/OneDrive/Bureau/kaggleBirds/wav2vec/').to('cuda:0')


def wav2vec_processing(mode):
    path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/'
    saving_path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/'

    batch_size = 64
    # # n_subtracks = 8762
    # n_loops = n_subtracks // batch_size
    # reste = n_subtracks % batch_size

    # mode = 'train/'
    path += mode
    l = os.listdir(path)
    saving_path += mode
    n_subtracks = len(l)
    n_loops = n_subtracks // batch_size
    reste = n_subtracks % batch_size
    # i = 0

    # print('model device')
    # print(model.device)

    for k in range(n_loops):
        t = torch.load(path + l[0 + batch_size*k], map_location='cuda:0')
        for i in range(batch_size - 1):
            x = torch.load(path + l[i + 1 + batch_size*k], map_location='cuda:0')
            t = torch.cat((t,x),dim=0)
        # print('tensor device')
        # print(t.device)
        # # print(t.shape)
        # end1 = time.time()
        # print('loading')
        # print(end1 - start)

        # start = time.time()
        with torch.no_grad():
            input_values = processor(t, return_tensors="pt",sampling_rate = 16000).input_values.to('cuda:0')  # Batch size 1
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
        t = torch.load(path + l[done], map_location='cuda:0')
        for i in range(reste - 1):
            x = torch.load(path + l[i + 1 + done], map_location='cuda:0')
            t = torch.cat((t,x),dim=0)
        # print(t.shape)

        with torch.no_grad():
            input_values = processor(t, return_tensors="pt",sampling_rate = 16000).input_values.to('cuda:0')  # Batch size 1
            # print(input_values.shape)

            # print('ok')
            # print(input_values.squeeze(0).shape)
            hidden_states = model(input_values.squeeze(0)).last_hidden_state
            # print(hidden_states.shape)

        for t in range(reste):
            torch.save(hidden_states[t], saving_path + l[t + done])
        # torch.save(hidden_states, saving_path + l[done])
    shutil.rmtree(path)
    os.makedirs(path)

# class LSTM(nn.Module):
#     """
#     input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
#     hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
#     output_size - This will be equal to the prediciton_periods input to get_x_y_pairs
#     """
#     def __init__(self, input_size, hidden_size, output_size):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
        
#         self.lstm = nn.LSTM(input_size, hidden_size,num_layers = 1)
        
#         self.linear = nn.Linear(hidden_size, output_size)

#         # self.linear2 = nn.Linear(output_size, 12)

#         # self.relu = nn.ReLU()
        
#     def forward(self, x, hidden=None):
#         if hidden==None:
#             self.hidden = (torch.zeros(1,1,self.hidden_size).to('cuda:0'),
#                            torch.zeros(1,1,self.hidden_size).to('cuda:0'))
#         else:
#             self.hidden = hidden
            
#         """
#         inputs need to be in the right shape as defined in documentation
#         - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        
#         lstm_out - will contain the hidden states from all times in the sequence
#         self.hidden - will contain the current hidden state and cell state
#         """
#         lstm_out, self.hidden = self.lstm(x.view(len(x),1,-1), 
#                                           self.hidden)
#         # print(x.shape)
#         # print('ee')
#         # print(lstm_out.shape)
#         # print('len')
#         # print(len(x))
#         # print('lstm_out.view(len(x), -1)')
#         # print(lstm_out.view(len(x), -1).shape)
#         predictions = self.linear(lstm_out.view(len(x), -1))
#         # predictions = self.relu(predictions)
#         # predictions = self.linear2(predictions)
#         # print('pred')
#         # print(predictions.shape)
#         # print('-1')
#         # print(predictions[-1].shape)
        
#         return predictions[-1], self.hidden

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
            self.hidden = (torch.zeros(1,batch_size,self.hidden_size).to('cuda:0'),
                           torch.zeros(1,batch_size,self.hidden_size).to('cuda:0'))
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

def pathToSpecies(path,dic):
    return dic[path.split('_')[2]]

# def trainer(quelbatch,n_list,epochs=2):

#     path_train = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/train/'
#     path_test = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/test/'
#     path_calib = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/calib/'
#     saving_path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/batch' + str(quelbatch) + '/nlist_' + str(n_list)

#     l_train = os.listdir(path_train)
#     l_test = os.listdir(path_test)
#     l_calib = os.listdir(path_calib)

#     wc = np.zeros(12)
#     for file in l_test:
#         wc[pathToSpecies(file,species_dict)] += 1
#     with open('infos.txt', 'a') as f:
#             f.write('test_sizes : ')
#             f.write('\n')
#             f.write( np.array_str(wc))
#             f.write('\n')
#     # print(wc)
#     for file in l_calib:
#         wc[pathToSpecies(file,species_dict)] += 1
#     with open('infos.txt', 'a') as f:
#             f.write('calib_sizes : ')
#             f.write('\n')
#             f.write(np.array_str(wc))
#             f.write('\n')
#     # print(wc)
#     for file in l_train:
#         wc[pathToSpecies(file,species_dict)] += 1
#     with open('infos.txt', 'a') as f:
#             f.write('train_sizes : ')
#             f.write('\n')
#             f.write(np.array_str(wc))
#             f.write('\n')
#     # print(wc)
#     for i,ele in enumerate(wc):
#         wc[i] = 1/ele
#     w = torch.from_numpy(wc).float().to('cuda:0')

#     model = LSTM(input_size=768, hidden_size=256, output_size=12).to('cuda:0')
#     criterion = nn.CrossEntropyLoss()
#     # criterion = nn.CrossEntropyLoss(weight = w)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

#     # epochs = 10

#     for epoch in range(epochs):
#         model.train()
#         start = time.time()
#         # for x,y in zip(x_train, y_train):
#         # for k in df_train.itertuples():
#         #     filename = t + str(k.Index) + '.hdf5'
#         #     with h5py.File(filename, "r") as f:
#         #         a_group_key = list(f.keys())[0]
#         #         x = torch.from_numpy(f[a_group_key][()]).squeeze(0)
#         #         # print(x.shape)
#         random.shuffle(l_train)
#         for file in l_train:
#             t = torch.load(path_train + file, map_location='cuda:0')
#             # print(path + file)
#             # print(t.get_device())
#             y_hat, _ = model(t, None)
#             # y = to_categorical(species_dict[x.primary_label],num_classes = 9)
#             # print(file)
#             # print(pathToSpecies(file,species_dict))
#             y = torch.as_tensor(pathToSpecies(file,species_dict)).to('cuda:0')
#             # print(y.shape)
#             optimizer.zero_grad()
#             # print(y_hat.shape)
#             loss = criterion(y_hat, y)
#             loss.backward()
#             optimizer.step()
            
#         if epoch%1==0:
#             print(f'epoch: {epoch:4} loss:{loss.item():10.8f}')
#         end = time.time()
#         print('time : ',end - start)
#         # scheduler.step()

#         model.eval()
#         acc = 0
#         with torch.no_grad():
#             for file in l_test:
#                 t = torch.load(path_test + file, map_location='cuda:0')
#                 # print(path + file)
#                 # print(t.get_device())
#                 y_hat, _ = model(t, None)
#                 # y = to_categorical(species_dict[x.primary_label],num_classes = 9)
#                 # print(file)
#                 # print(pathToSpecies(file,species_dict))
#                 # print(y_hat)
#                 # print(torch.argmax(y_hat))
#                 # print(pathToSpecies(file,species_dict))
#                 acc += ( torch.argmax(y_hat) == pathToSpecies(file,species_dict) )
#                 # y = torch.as_tensor(pathToSpecies(file,species_dict)).to('cuda:0')
#         print('accuracy : ',acc/len(l_test))
#         if epoch == 9:
#             with open('infos.txt', 'a') as f:
#                 f.write('accuracy : ' +  str(acc/len(l_test)))
#                 f.write('\n')
#     torch.save(model.state_dict(), saving_path)
#     return model

def trainer(quelbatch,n_list,epochs=10):

    path_train = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/train/'
    path_test = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/test/'
    path_calib = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/calib/'
    saving_path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/batch' + str(quelbatch) + '/nlist_' + str(n_list)

    l_train = os.listdir(path_train)
    l_test = os.listdir(path_test)
    l_calib = os.listdir(path_calib)

    wc = np.zeros(12)
    for file in l_test:
        wc[pathToSpecies(file,species_dict)] += 1
    with open('infos.txt', 'a') as f:
            f.write('test_sizes : ')
            f.write('\n')
            f.write( np.array_str(wc))
            f.write('\n')
    # print(wc)
    for file in l_calib:
        wc[pathToSpecies(file,species_dict)] += 1
    with open('infos.txt', 'a') as f:
            f.write('calib_sizes : ')
            f.write('\n')
            f.write(np.array_str(wc))
            f.write('\n')
    # print(wc)
    for file in l_train:
        wc[pathToSpecies(file,species_dict)] += 1
    with open('infos.txt', 'a') as f:
            f.write('train_sizes : ')
            f.write('\n')
            f.write(np.array_str(wc))
            f.write('\n')
    # print(wc)
    for i,ele in enumerate(wc):
        wc[i] = 1/ele
    w = torch.from_numpy(wc).float().to('cuda:0')

    model = LSTM(input_size=768, hidden_size=256, output_size=12).to('cuda:0')
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight = w)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                   step_size = 8, # Period of learning rate decay
                   gamma = 0.5) # Multiplicative factor of learning rate decay
    batch_size = 4

    for epoch in range(epochs):
        model.train()
        start = time.time()
        # for x,y in zip(x_train, y_train):
        # for k in df_train.itertuples():
        #     filename = t + str(k.Index) + '.hdf5'
        #     with h5py.File(filename, "r") as f:
        #         a_group_key = list(f.keys())[0]
        #         x = torch.from_numpy(f[a_group_key][()]).squeeze(0)
        #         # print(x.shape)
        random.shuffle(l_train)
        nelem = len(l_train)
        n_loops = nelem // batch_size
        reste = nelem % batch_size
        # for file in l_train:
        for j in range(n_loops):
            t = torch.load(path_train + l_train[j*batch_size], map_location='cuda:0').unsqueeze(dim=0)
            # print('a')
            # print(t.shape)
            y = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_train[j*batch_size],species_dict)
            for i in range(batch_size - 1):
                x = torch.load(path_train + l_train[j*batch_size + i + 1], map_location='cuda:0').unsqueeze(dim=0)
                z = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_train[j*batch_size + i + 1],species_dict)
                t = torch.cat((t,x),dim=0)
                # print('b')
                # print(t.shape)
                y = torch.cat((y,z),dim=0)
            y_hat, _ = model(t, batch_size, None)
            optimizer.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
        if reste != 0:
            done = batch_size * n_loops
            t = torch.load(path_train + l_train[done], map_location='cuda:0').unsqueeze(dim=0)
            # y = torch.as_tensor(pathToSpecies(l_train[done],species_dict)).to('cuda:0')
            y = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_train[done],species_dict)
            for i in range(reste - 1):
                x = torch.load(path_train + l_train[done + i + 1], map_location='cuda:0').unsqueeze(dim=0)
                z = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_train[done + i + 1],species_dict)
                t = torch.cat((t,x),dim=0)
                y = torch.cat((y,z),dim=0)
            y_hat, _ = model(t, reste, None)
            optimizer.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

        if epoch%1==0:
            print(f'epoch: {epoch:4} loss:{loss.item():10.8f}')
        end = time.time()
        print('time : ',end - start)
        scheduler.step()

        model.eval()
        acc = 0
        with torch.no_grad():
            # for file in l_test:
            #     t = torch.load(path_test + file, map_location='cuda:0')
            #     # print(path + file)
            #     # print(t.get_device())
            #     y_hat, _ = model(t, batch_sizeNone)
            #     # y = to_categorical(species_dict[x.primary_label],num_classes = 9)
            #     # print(file)
            #     # print(pathToSpecies(file,species_dict))
            #     # print(y_hat)
            #     # print(torch.argmax(y_hat))
            #     # print(pathToSpecies(file,species_dict))
            #     acc += ( torch.argmax(y_hat) == pathToSpecies(file,species_dict) )
            #     # y = torch.as_tensor(pathToSpecies(file,species_dict)).to('cuda:0')
                
            nelem = len(l_test)
            n_loops = nelem // batch_size
            reste = nelem % batch_size
            # for file in l_train:
            for j in range(n_loops):
                t = torch.load(path_test + l_test[j*batch_size], map_location='cuda:0').unsqueeze(dim=0)
                # print('a')
                # print(t.shape)
                y = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_test[j*batch_size],species_dict)
                for i in range(batch_size - 1):
                    x = torch.load(path_test + l_test[j*batch_size + i + 1], map_location='cuda:0').unsqueeze(dim=0)
                    z = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_test[j*batch_size + i + 1],species_dict)
                    t = torch.cat((t,x),dim=0)
                    # print('b')
                    # print(t.shape)
                    y = torch.cat((y,z),dim=0)
                y_hat, _ = model(t, batch_size, None)
                # optimizer.zero_grad()
                # loss = criterion(y_hat, y)
                # loss.backward()
                # optimizer.step()
                # print('y_hat')
                # print(y_hat.shape)
                for k in range(y_hat.shape[0]):
                    acc += ( torch.argmax(y_hat[k]) == y[k] )
            if reste != 0:
                done = batch_size * n_loops
                t = torch.load(path_test + l_test[done], map_location='cuda:0').unsqueeze(dim=0)
                # y = torch.as_tensor(pathToSpecies(l_train[done],species_dict)).to('cuda:0')
                y = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_test[done],species_dict)
                for i in range(reste - 1):
                    x = torch.load(path_test + l_test[done + i + 1], map_location='cuda:0').unsqueeze(dim=0)
                    z = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_test[done + i + 1],species_dict)
                    t = torch.cat((t,x),dim=0)
                    y = torch.cat((y,z),dim=0)
                y_hat, _ = model(t, reste, None)
                # optimizer.zero_grad()
                # loss = criterion(y_hat, y)
                # loss.backward()
                # optimizer.step()
                for k in range(y_hat.shape[0]):
                    acc += ( torch.argmax(y_hat[k]) == y[k] )


        print('accuracy : ',acc/len(l_test))
        if epoch == epochs - 1:
            with open('infos.txt', 'a') as f:
                f.write('accuracy : ' +  str(acc/len(l_test)))
                f.write('\n')
    torch.save(model.state_dict(), saving_path)
    return model


### SPECIAL

# mode = 'train/'
# path += mode
# l = os.listdir('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/' + mode )
# saving_path += mode
# n_subtracks = len(l)
# n_loops = n_subtracks // batch_size
# reste = n_subtracks % batch_size

    
### FIN WAV2VEC MODEL ###

### DEBUT TRAINING ###

### FIN TRAINING ###

### DEBUT CP ###

### FIN CP ###

### MAIN ### infos generales --> extraction (stats)--> training (models) --> infos CP --> inference (choix de l heuristique)

### DEBUT MAIN ###



### DEBUT EXTRACTION ### PREND UNE LISTE D ESPECES AVEC OPTIONS ? ---> FICHIERS + STATS ? 

def ratingfn(n,batch):
    if n < 7 and batch == 0:
        return 3
    elif n < 3 and batch == 1:
        return 3
    else:
        return 0
    
def select_training_list(species,training_list,rating,n_list,batch):
            return species in training_list and rating > ratingfn(n_list,batch)


def SelectIndexes(df,dic):
    listIndexes = []
    Indexes = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],
                7:[],8:[],9:[],10:[],11:[]}
    for x in df.itertuples():
        Indexes[dic[x.primary_label]].append(x.Index)
    # print(Indexes)
    for key in list(Indexes.keys()):
        random.shuffle(Indexes[key])
    l = list(range(12))
    while Indexes != {}:
        for ind,i in enumerate(l):
            if Indexes[i] != []:
                x = Indexes[i].pop()
                listIndexes.append(x)
            else:
                Indexes.pop(i)
                del(l[ind])
    return listIndexes

with open('infos.txt', 'w') as f:
    f.write('essai 12 birds')
    f.write('\n')

steps_per_subtrack = 32000
flush_every = 200
limit = 14000

transform = torchaudio.transforms.Resample(32000, 16000)
path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/train_audio/'

train_metadata = pd.read_csv('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/train_metadata.csv')

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

batch1 = {}
batch2 = {}
for j in range(11):
    batch2['list_' + str(2*j)] = [sorted_list[12*2*j] ,sorted_list[12*2*j + 2] ,sorted_list[12*2*j + 4] ,sorted_list[12*2*j +6] ,
                                sorted_list[12*2*j + 8] ,sorted_list[12*2*j + 10] ,sorted_list[12*2*j + 12] ,sorted_list[12*2*j + 14] ,
                                sorted_list[12*2*j + 16] ,sorted_list[12*2*j + 18] ,sorted_list[12*2*j + 20] ,sorted_list[12*2*j + 22]  ]
    
    batch2['list_' + str(2*j + 1)] =  [sorted_list[12*2*j + 1] ,sorted_list[12*2*j + 3] ,sorted_list[12*j + 5] ,sorted_list[12*2*j +7] ,
                                sorted_list[12*2*j + 9] ,sorted_list[12*2*j + 11] ,sorted_list[12*2*j + 13] ,sorted_list[12*2*j + 15] ,
                                sorted_list[12*2*j + 17] ,sorted_list[12*2*j + 19] ,sorted_list[12*2*j + 21] ,sorted_list[12*2*j + 23]  ]
    
    batch1['list_' + str(2*j)] = [sorted_list[12*j] ,sorted_list[12*j + 1] ,sorted_list[12*j + 2] ,sorted_list[12*j +3] ,
                                sorted_list[12*j + 4] ,sorted_list[12*j + 5] ,sorted_list[12*j + 6] ,sorted_list[12*j + 7] ,
                                sorted_list[12*j + 8] ,sorted_list[12*j + 9] ,sorted_list[12*j + 10] ,sorted_list[12*j + 11]  ]
    
    batch1['list_' + str(2*j + 1)] = [sorted_list[12*(j+1)] ,sorted_list[12*(j+1) + 1] ,sorted_list[12*(j+1) + 2] ,sorted_list[12*(j+1) +3] ,
                                sorted_list[12*(j+1) + 4] ,sorted_list[12*(j+1) + 5] ,sorted_list[12*(j+1) + 6] ,sorted_list[12*(j+1) + 7] ,
                                sorted_list[12*(j+1) + 8] ,sorted_list[12*(j+1) + 9] ,sorted_list[12*(j+1) + 10] ,sorted_list[12*(j+1) + 11]  ]

for quelbatch in range(2):
    for n_list in range(22):
        # quelbatch = 1
        # n_list = 21
        # special_rating = 0

        # n_list = 21
        if quelbatch == 0 and n_list == 2:
            flush_every = 50
        else:
            flush_every = 150
        test_ind = []
        used_ind = []

        if (quelbatch == 0 and (n_list < 9)):
        # if False:
            print('avoided')
        else:
            print('batch : ',quelbatch,' n_list : ',n_list)
            with open('infos.txt', 'a') as f:
                f.write('batch : ' + str(quelbatch) +  ' n_list : ' +  str(n_list))
                f.write('\n')
            

            os.makedirs('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/train/')
            os.makedirs('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/test/')
            os.makedirs('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/calib/')
            os.makedirs('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/train/')
            os.makedirs('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/test/')
            os.makedirs('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/calib/')

            
            if quelbatch  + 1 == 1:
                batch = batch1
            else: 
                batch = batch2
                
            training_list = batch['list_' + str(n_list)]
            species_dict = {}
            for i,species in enumerate(training_list):
                species_dict[species] = i
            
            train_metadata = pd.read_csv('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/train_metadata.csv')

            train_metadata.insert(column='selected',loc=1,value=int)
            for x in train_metadata.itertuples():
                train_metadata.at[x.Index,'selected'] = select_training_list(x.primary_label,species_dict,x.rating,n_list,quelbatch)

            train = train_metadata[train_metadata['selected'] == 1].sample(frac=0.7,random_state=0)
            test_calib = train_metadata[train_metadata['selected'] == 1].drop(train.index)
            test = test_calib.sample(frac=0.5,random_state=0)
            calib = test_calib.drop(test.index)

            train_metadata.insert(column='ttc',loc=2,value=str)
            for x in train_metadata.itertuples():
                if x.Index in train.index:
                    train_metadata.at[x.Index,'tort'] = 'train'
                elif x.Index in test.index:
                    train_metadata.at[x.Index,'tort'] = 'test'
                elif x.Index in calib.index:
                    train_metadata.at[x.Index,'tort'] = 'calib'

            k = 0

            path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/train_audio/'

            processed = 0
            processed_all = 0
            start = time.time()

            # train_metadata = train_metadata.sample(frac=1).reset_index(drop=True) 
            Indexes = SelectIndexes(train_metadata[train_metadata['selected'] == 1],species_dict)
            # for x in train_metadata[train_metadata['selected'] == 1].itertuples():
            for ind in Indexes:
                # pl = x.primary_label
                # sl = x.secondary_labels
                # rating = str(x.rating)
                # type = x.type
                # audio_index = str(x.Index)
                # # print(index)
                # id = x.filename
                used_ind.append(ind)

                pl = train_metadata['primary_label'][ind]
                sl = train_metadata['secondary_labels'][ind]
                rating = str(train_metadata['rating'][ind])
                type = train_metadata['type'][ind]
                audio_index = str(ind)
                # print(index)
                id = train_metadata['filename'][ind]
                tort = train_metadata['tort'][ind]

                file = path + id
                data, samplerate = torchaudio.load(file, normalize=True)
                data = transform(data)
                track_length = data.shape[1]
                tl = str(track_length)
                
                # if x.tort == 'train':
                if tort == 'train':
                    for i in range(track_length // steps_per_subtrack):
                        # with torch.no_grad():
                            # input_values = processor(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], return_tensors="pt",sampling_rate = 16000).input_values.to('cuda:0')  # Batch size 1
                            # hidden_states = model(input_values.squeeze(0)).last_hidden_state
                            # subtrack_index = str(k)
                            # torch.save(hidden_states, 'data/subtracks64000/train/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
                            # k += 1
                        subtrack_index = str(k)
                        torch.save(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], 'data/subtracks64000/train/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
                        k += 1
                        processed += 1
                # elif x.tort == 'test':
                elif tort == 'test':
                    for i in range(track_length // steps_per_subtrack):
                        # with torch.no_grad():
                            # input_values = processor(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], return_tensors="pt",sampling_rate = 16000).input_values.to('cuda:0')  # Batch size 1
                            # hidden_states = model(input_values.squeeze(0)).last_hidden_state
                            # subtrack_index = str(k)
                            # torch.save(hidden_states, 'data/subtracks64000/test/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
                            # k += 1
                        subtrack_index = str(k)
                        torch.save(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], 'data/subtracks64000/test/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
                        k += 1
                        processed += 1
                    test_ind.append(ind)
                # elif x.tort == 'calib':
                elif tort == 'calib':
                    for i in range(track_length // steps_per_subtrack):
                        # with torch.no_grad():
                            # input_values = processor(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], return_tensors="pt",sampling_rate = 16000).input_values.to('cuda:0')  # Batch size 1
                            # hidden_states = model(input_values.squeeze(0)).last_hidden_state
                            # subtrack_index = str(k)
                            # torch.save(hidden_states, 'data/subtracks64000/calib/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
                            # k += 1
                        subtrack_index = str(k)
                        torch.save(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], 'data/subtracks64000/calib/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
                        k += 1
                        processed += 1
                
                
                
                if (processed > flush_every ):
                    processed_all += processed
                    processed = 0
                    # print(processed_all)
                    # end = time.time()
                    # print('x')
                    # print(end - start)
                    # start = time.time()
                    wav2vec_processing('train/')
                    # print('x_traon')
                    # end = time.time()
                    # print(end - start)
                    # start = time.time()
                    wav2vec_processing('test/')
                    # print('x_test')
                    # end = time.time()
                    # print(end - start)
                    wav2vec_processing('calib/') 

                if (processed_all > limit):
                    limit_reached = True
                    break

            if processed_all + processed < limit :
                wav2vec_processing('train/')
                wav2vec_processing('test/')
                wav2vec_processing('calib/')
                shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/train/')
                shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/test/')
                shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/calib/')
            else:
                shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/train/')
                shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/test/')
                shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/calib/')

            ### export information
            ### train
            end = time.time()
            print('process ended : ')
            print(end-start)
            model_class = trainer(quelbatch + 1,n_list)
            # shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/train/')
            # shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/test/')
            # shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/calib/')
            with open('infos.txt', 'a') as f:
                f.write('test indices')
                f.write('\n')
                q = ' '.join(str(e) for e in test_ind)
                f.write(q)
                f.write('\n')
                f.write('used indices')
                f.write('\n')
                q = ' '.join(str(e) for e in used_ind)
                f.write(q)
                f.write('\n')
            ### calibrate
            calibrate(model_class,species_dict,[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8])
            shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/train/')
            shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/test/')
            shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/calib/')
            ### export information
