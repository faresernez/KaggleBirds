import torch.nn as nn
from torch import optim
import pandas as pd
import torchaudio
import torch
torch.backends.cudnn.benchmark = True
import os
from datasets import load_dataset
from pathlib import Path
import re
import json
from transformers import AutoConfig , AutoTokenizer , AutoFeatureExtractor , Wav2Vec2Model , Wav2Vec2Processor
torch.cuda.empty_cache()
import time
import random
import numpy as np

class LSTM(nn.Module):
    """
    input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediciton_periods input to get_x_y_pairs
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers = 2)
        
        self.linear = nn.Linear(hidden_size, output_size)

        
        
    def forward(self, x, hidden=None):
        if hidden==None:
            self.hidden = (torch.zeros(2,1,self.hidden_size).to('cuda:0'),
                           torch.zeros(2,1,self.hidden_size).to('cuda:0'))
        else:
            self.hidden = hidden
            
        """
        inputs need to be in the right shape as defined in documentation
        - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        
        lstm_out - will contain the hidden states from all times in the sequence
        self.hidden - will contain the current hidden state and cell state
        """
        lstm_out, self.hidden = self.lstm(x.view(len(x),1,-1), 
                                          self.hidden)
        # print(x.shape)
        # print('ee')
        # print(lstm_out.shape)
        # print('len')
        # print(len(x))
        # print('lstm_out.view(len(x), -1)')
        # print(lstm_out.view(len(x), -1).shape)
        predictions = self.linear(lstm_out.view(len(x), -1))
        predictions = self.linear2(predictions)
        # print('pred')
        # print(predictions.shape)
        # print('-1')
        # print(predictions[-1].shape)
        
        return predictions[-1], self.hidden

species_dict = {}
size_dict = {}
pathlist = Path(r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\train_audio').glob('*')
i = 0
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    species = path_in_str[60:]
    species_dict[species] = i
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
# print(batch1)
# print(batch2)

batch = batch1
n_list = 5
training_list = batch['list_' + str(n_list)]
species_dict = {}
for i,species in enumerate(training_list):
    species_dict[species] = i

# print(converted_dict)
# print(size_dict)
path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/'
# mode = 'train/'= mode
path_train = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/train/'
# mode = 'train/'
# path += mode
# l = os.listdir('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/' + mode )
l = os.listdir(path_train)

path_test = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/test/'
l_test = os.listdir(path_test)

def pathToSpecies(path,dic):
    return dic[path.split('_')[2]]

wc = np.zeros(12)
for file in l:
    wc[pathToSpecies(file,species_dict)] += 1
print(wc)
for i,ele in enumerate(wc):
    wc[i] = 1/ele
w = torch.from_numpy(wc).float().to('cuda:0')
print(w)

print(species_dict)


model = LSTM(input_size=768, hidden_size=512, output_size=12).to('cuda:0')
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss(weight = w)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

epochs = 600

for epoch in range(epochs+1):
    model.train()
    start = time.time()
    # for x,y in zip(x_train, y_train):
    # for k in df_train.itertuples():
    #     filename = t + str(k.Index) + '.hdf5'
    #     with h5py.File(filename, "r") as f:
    #         a_group_key = list(f.keys())[0]
    #         x = torch.from_numpy(f[a_group_key][()]).squeeze(0)
    #         # print(x.shape)
    random.shuffle(l)
    for file in l:
        t = torch.load(path_train + file, map_location='cuda:0')
        # print(path + file)
        # print(t.get_device())
        y_hat, _ = model(t, None)
        # y = to_categorical(species_dict[x.primary_label],num_classes = 9)
        # print(file)
        # print(pathToSpecies(file,species_dict))
        y = torch.as_tensor(pathToSpecies(file,species_dict)).to('cuda:0')
        # print(y.shape)
        optimizer.zero_grad()
        # print(y_hat.shape)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        
    if epoch%1==0:
        print(f'epoch: {epoch:4} loss:{loss.item():10.8f}')
    end = time.time()
    print('time : ',end - start)
    # scheduler.step()

    model.eval()
    acc = 0
    with torch.no_grad():
        for file in l_test:
            t = torch.load(path_test + file, map_location='cuda:0')
            # print(path + file)
            # print(t.get_device())
            y_hat, _ = model(t, None)
            # y = to_categorical(species_dict[x.primary_label],num_classes = 9)
            # print(file)
            # print(pathToSpecies(file,species_dict))
            # print(y_hat)
            # print(torch.argmax(y_hat))
            # print(pathToSpecies(file,species_dict))
            acc += ( torch.argmax(y_hat) == pathToSpecies(file,species_dict) )
            # y = torch.as_tensor(pathToSpecies(file,species_dict)).to('cuda:0')
    print('accuracy : ',acc/len(l_test))