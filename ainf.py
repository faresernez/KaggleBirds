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

# q_hat = []
q_hat = 0.2

path_test = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/infer/hidden64000/'
l_test = os.listdir(path_test)

batch_size = 64
nelem = len(l_test)
print(nelem)
n_loops = nelem // batch_size
reste = nelem % batch_size



model = LSTM(input_size=768, hidden_size=256, output_size=12).to('cuda:0')
model.load_state_dict(torch.load(r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\models\batch1\nlist_0'))
model.eval()

with torch.no_grad():
    for j in range(n_loops):
        t = torch.load(path_test + l_test[j*batch_size], map_location='cuda:0').unsqueeze(dim=0)
        # y = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_test[j*batch_size],species_dict)
        for i in range(batch_size - 1):
                x = torch.load(path_test + l_test[j*batch_size + i + 1], map_location='cuda:0').unsqueeze(dim=0)
                # z = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_test[j*batch_size + i + 1],species_dict)
                t = torch.cat((t,x),dim=0)
                # print('b')
                # print(t.shape)
                # y = torch.cat((y,z),dim=0)
        y_hat, _ = model(t, batch_size, None)
        if j == 0:
            ypred_test = y_hat
            # yt_test = y
        else:
            ypred_test = torch.cat((ypred_test,y_hat),dim=0)
            # yt_test = torch.cat((yt_test,y),dim=0)
    if reste != 0:
        done = batch_size * n_loops
        t = torch.load(path_test + l_test[done], map_location='cuda:0').unsqueeze(dim=0)
        # y = torch.as_tensor(pathToSpecies(l_train[done],species_dict)).to('cuda:0')
        # y = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_test[done],species_dict)
        for i in range(reste - 1):
            x = torch.load(path_test + l_test[done + i + 1], map_location='cuda:0').unsqueeze(dim=0)
            # z = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_test[done + i + 1],species_dict)
            t = torch.cat((t,x),dim=0)
            # y = torch.cat((y,z),dim=0)
        y_hat, _ = model(t, reste, None)
        ypred_test = torch.cat((ypred_test,y_hat),dim=0)
        # yt_test = torch.cat((yt_test,y),dim=0).cpu().numpy()
        ypred_test.cpu().numpy()
# print(ypred_test.shape)
all_pred = []
for i,elem in enumerate(ypred_test):
    pred = []
    for classe,soft in enumerate(elem):
        if (1 - soft) <= q_hat:
            pred.append(classe)
    all_pred.append(pred)