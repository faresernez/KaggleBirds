import time
import pandas as pd
import glob
import os
import torch
import pandas as pd
import os
import numpy as np
from pathlib import Path
import torchaudio
import torch
import shutil
import random
import torch.nn as nn
from torch import optim
import librosa
torch.backends.cudnn.benchmark = True
import copy
from torch.ao.quantization import get_default_qconfig, get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torchvision.models import resnet50
from code2023.calibration2 import calibratev3
from code2023.calibration import calibratev2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# class cinnv3(nn.Module):
#     # def __init__(self, input_size, hidden_size, output_size):
#     def __init__(self):
#         super(cinnv3, self).__init__()
#         self.relu = nn.ReLU()
#         self.cinn1 = nn.Conv2d(1, 64, 5, stride=2, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.cinn2 = nn.Conv2d(64, 128, 5, stride=2, padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.cinn3 = nn.Conv2d(128, 256, 5, stride=2, padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.cinn4 = nn.Conv2d(256, 256, 3, stride=3, padding=0, bias=False)
#         self.mxp = nn.MaxPool2d(4, stride=2)
#         self.lin = nn.Linear(in_features=256, out_features=24)
#         self.soft = nn.Softmax(dim=1)

#     def forward(self,x):
#         x = self.cinn1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.cinn2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.cinn3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.cinn4(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.mxp(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.lin(x)
#         x = self.soft(x)
#         return x

# class cinnv2(nn.Module):
#     # def __init__(self, input_size, hidden_size, output_size):
#     def __init__(self):
#         super(cinnv2, self).__init__()
#         self.relu = nn.ReLU()
#         self.cinn1 = nn.Conv2d(1, 64, 5, stride=2, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.cinn2 = nn.Conv2d(64, 128, 5, stride=2, padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.cinn3 = nn.Conv2d(128, 256, 5, stride=2, padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.cinn4 = nn.Conv2d(256, 256, 3, stride=3, padding=0, bias=False)
#         self.mxp = nn.MaxPool2d(4, stride=2)
#         self.lin = nn.Linear(in_features=256, out_features=12)
#         self.soft = nn.Softmax(dim=1)

#     def forward(self,x):
#         x = self.cinn1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.cinn2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.cinn3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.cinn4(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.mxp(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.lin(x)
#         x = self.soft(x)
#         return x

class cinnv3(nn.Module):
    # def __init__(self, input_size, hidden_size, output_size):
    def __init__(self):
        super(cinnv3, self).__init__()
        self.relu = nn.ReLU()
        self.cinn1 = nn.Conv2d(1, 64, 3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.cinn2 = nn.Conv2d(64, 128, 3, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.cinn3 = nn.Conv2d(128, 128, 3, stride=2, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(64)
        self.cinn4 = nn.Conv2d(128, 128, 3, stride=3, padding=0, bias=False)
        self.mxp = nn.MaxPool2d(4, stride=2)
        self.lin = nn.Linear(in_features=128, out_features=24)
        self.soft = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.cinn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.cinn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.cinn3(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.cinn4(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.mxp(x)
        x = torch.flatten(x, start_dim=1)
        x = self.lin(x)
        x = self.soft(x)
        return x

class cinnv2(nn.Module):
    # def __init__(self, input_size, hidden_size, output_size):
    def __init__(self):
        super(cinnv2, self).__init__()
        self.relu = nn.ReLU()
        self.cinn1 = nn.Conv2d(1, 64, 3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.cinn2 = nn.Conv2d(64, 128, 3, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.cinn3 = nn.Conv2d(128, 128, 3, stride=2, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(64)
        self.cinn4 = nn.Conv2d(128, 128, 3, stride=3, padding=0, bias=False)
        self.mxp = nn.MaxPool2d(4, stride=2)
        self.lin = nn.Linear(in_features=128, out_features=12)
        self.soft = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.cinn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.cinn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.cinn3(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.cinn4(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.mxp(x)
        x = torch.flatten(x, start_dim=1)
        x = self.lin(x)
        x = self.soft(x)
        return x
    
def pathToSpecies(path,dic):
    return dic[path.split('_')[2]]

def ratingfn(n,batch):
    if n < 7 and batch == 0:
        return 0
    elif n < 3 and batch == 1:
        return 0
    else:
        return 0
    
def select_training_list(species,training_list,rating,n_list,batch):
    return species in training_list 
            # return species in training_list and rating > ratingfn(n_list,batch)


def SelectIndexes(df,dic):
    listIndexes = []
    Indexes = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],
                7:[],8:[],9:[],10:[],11:[],12:[],13:[],
                14:[],15:[],16:[],17:[],18:[],19:[],20:[],
                21:[],22:[],23:[]}
    for x in df.itertuples():
        Indexes[dic[x.primary_label]].append(x.Index)
    # print(Indexes)
    for key in list(Indexes.keys()):
        random.shuffle(Indexes[key])
    l = list(range(24))
    while Indexes != {}:
        for ind,i in enumerate(l):
            if Indexes[i] != []:
                x = Indexes[i].pop()
                listIndexes.append(x)
            else:
                Indexes.pop(i)
                del(l[ind])
    return listIndexes

def epp(n_list):
    if n_list < 2:
        return 5
    elif n_list > 8:
        return 15
    else:
        return 7

def trainer(quelbatch,n_list,epochs=1):

    path_train = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/train/'
    path_test = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/test/'
    path_calib = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/calib/'
    saving_path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/batch' + str(quelbatch) + '/nlist_' + str(n_list)
    model_path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/model'

    l_train = os.listdir(path_train)
    l_test = os.listdir(path_test)
    l_calib = os.listdir(path_calib)

    if n_list < 10:
        wc = np.zeros(24)
    else:
        wc = np.zeros(12)
    # for file in l_test:
    #     wc[pathToSpecies(file,species_dict)] += 1
    # with open('infos.txt', 'a') as f:
    #         f.write('test_sizes : ')
    #         f.write('\n')
    #         f.write( np.array_str(wc))
    #         f.write('\n')
    # # print(wc)
    # for file in l_calib:
    #     wc[pathToSpecies(file,species_dict)] += 1
    # with open('infos.txt', 'a') as f:
    #         f.write('calib_sizes : ')
    #         f.write('\n')
    #         f.write(np.array_str(wc))
    #         f.write('\n')
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
    
    if n_list < 10:
        model = cinnv3().to('cuda:0')
        model.load_state_dict(torch.load(model_path))
        # model = nn.Sequential(*list(modelx.children())[:-2],nn.Linear(in_features=128, out_features=24),
        #         nn.Softmax(dim=1)).to('cuda:0')
    else:
        model = cinnv3().to('cuda:0')
        # model = cinnv2().to('cuda:0')
        model.load_state_dict(torch.load(model_path))
        # model = nn.Sequential(*list(modelx.children())[:-2],nn.Linear(in_features=128, out_features=12),
        #         nn.Softmax(dim=1)).to('cuda:0')
        model._modules['lin'] = nn.Linear(in_features=128, out_features=12)
        model.to('cuda:0')

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight = w)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                   step_size = 8, # Period of learning rate decay
                   gamma = 0.5) # Multiplicative factor of learning rate decay
    batch_size = 4

    for epoch in range(epp(n_list)):
        model.train()
        start = time.time()
        random.shuffle(l_train)
        nelem = len(l_train)
        n_loops = nelem // batch_size
        reste = nelem % batch_size
        # for file in l_train:
        for j in range(n_loops):
            t = torch.load(path_train + l_train[j*batch_size], map_location='cuda:0').unsqueeze(dim=0)
            # print('shape')
            # print(t.shape)
            y = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_train[j*batch_size],species_dict)
            for i in range(batch_size - 1):
                x = torch.load(path_train + l_train[j*batch_size + i + 1], map_location='cuda:0').unsqueeze(dim=0)
                z = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_train[j*batch_size + i + 1],species_dict)
                t = torch.cat((t,x),dim=0)
                y = torch.cat((y,z),dim=0)
            # print('shape2')
            # print(t.shape)
            y_hat = model(t)
            # print(y_hat)
            # print(type(y_hat))
            # print(y_hat.shape)
            optimizer.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
        if reste != 0:
            done = batch_size * n_loops
            t = torch.load(path_train + l_train[done], map_location='cuda:0').unsqueeze(dim=0)
            y = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_train[done],species_dict)
            for i in range(reste - 1):
                x = torch.load(path_train + l_train[done + i + 1], map_location='cuda:0').unsqueeze(dim=0)
                z = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_train[done + i + 1],species_dict)
                t = torch.cat((t,x),dim=0)
                y = torch.cat((y,z),dim=0)
            y_hat = model(t)
            optimizer.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

        # if epoch%1==0:
        #     print(f'epoch: {epoch:4} loss:{loss.item():10.8f}')
        # end = time.time()
        # print('time : ',end - start)
        # scheduler.step()

        # model.eval()
        # acc = 0
        # with torch.no_grad():
                
        #     nelem = len(l_test)
        #     n_loops = nelem // batch_size
        #     reste = nelem % batch_size
        #     # for file in l_train:
        #     for j in range(n_loops):
        #         t = torch.load(path_test + l_test[j*batch_size], map_location='cuda:0').unsqueeze(dim=0)
        #         y = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_test[j*batch_size],species_dict)
        #         for i in range(batch_size - 1):
        #             x = torch.load(path_test + l_test[j*batch_size + i + 1], map_location='cuda:0').unsqueeze(dim=0)
        #             z = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_test[j*batch_size + i + 1],species_dict)
        #             t = torch.cat((t,x),dim=0)
        #             y = torch.cat((y,z),dim=0)
        #         y_hat = model(t)
        #         for k in range(y_hat.shape[0]):
        #             acc += ( torch.argmax(y_hat[k]) == y[k] )
        #     if reste != 0:
        #         done = batch_size * n_loops
        #         t = torch.load(path_test + l_test[done], map_location='cuda:0').unsqueeze(dim=0)
        #         y = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_test[done],species_dict)
        #         for i in range(reste - 1):
        #             x = torch.load(path_test + l_test[done + i + 1], map_location='cuda:0').unsqueeze(dim=0)
        #             z = torch.ones(1,device='cuda:0',dtype=torch.long)*pathToSpecies(l_test[done + i + 1],species_dict)
        #             t = torch.cat((t,x),dim=0)
        #             y = torch.cat((y,z),dim=0)
        #         y_hat = model(t)
        #         for k in range(y_hat.shape[0]):
        #             acc += ( torch.argmax(y_hat[k]) == y[k] )


        # print('accuracy : ',acc/len(l_test))
        # if epoch == epochs - 1:
        #     with open('infos.txt', 'a') as f:
        #         f.write('accuracy : ' +  str(acc/len(l_test)))
        #         f.write('\n')
    torch.save(model.state_dict(), saving_path)
    return model

with open('infos.txt', 'w') as f:
    f.write('essai 12 birds')
    f.write('\n')

steps_per_subtrack = 64000
flush_every = 200
limit = 40000 # 200000

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

batch0 = {}
batch1 = {}

# for j in range(11):
#     batch2['list_' + str(2*j)] = [sorted_list[12*2*j] ,sorted_list[12*2*j + 2] ,sorted_list[12*2*j + 4] ,sorted_list[12*2*j +6] ,
#                                 sorted_list[12*2*j + 8] ,sorted_list[12*2*j + 10] ,sorted_list[12*2*j + 12] ,sorted_list[12*2*j + 14] ,
#                                 sorted_list[12*2*j + 16] ,sorted_list[12*2*j + 18] ,sorted_list[12*2*j + 20] ,sorted_list[12*2*j + 22] ,
#                                   sorted_list[12*2*j + 24] ,sorted_list[12*2*j + 26] ,sorted_list[12*2*j + 28] ,sorted_list[12*2*j + 30] ,
#                                 sorted_list[12*2*j + 32] ,sorted_list[12*2*j + 34] ,sorted_list[12*2*j + 36] ,sorted_list[12*2*j + 38] ,
#                                 sorted_list[12*2*j + 40] ,sorted_list[12*2*j + 42] ,sorted_list[12*2*j + 44] ,sorted_list[12*2*j + 46]  ]
    
#     batch2['list_' + str(2*j + 1)] =  [sorted_list[12*2*j + 1] ,sorted_list[12*2*j + 3] ,sorted_list[12*j + 5] ,sorted_list[12*2*j +7] ,
#                                 sorted_list[12*2*j + 9] ,sorted_list[12*2*j + 11] ,sorted_list[12*2*j + 13] ,sorted_list[12*2*j + 15] ,
#                                 sorted_list[12*2*j + 17] ,sorted_list[12*2*j + 19] ,sorted_list[12*2*j + 21] ,sorted_list[12*2*j + 23] ,
#                                  sorted_list[12*2*j + 25] ,sorted_list[12*2*j + 27] ,sorted_list[12*j + 29] ,sorted_list[12*2*j + 31] ,
#                                 sorted_list[12*2*j + 33] ,sorted_list[12*2*j + 35] ,sorted_list[12*2*j + 37] ,sorted_list[12*2*j + 39] ,
#                                 sorted_list[12*2*j + 41] ,sorted_list[12*2*j + 43] ,sorted_list[12*2*j + 45] ,sorted_list[12*2*j + 47]  ]
    
#     batch1['list_' + str(2*j)] = [sorted_list[12*j] ,sorted_list[12*j + 1] ,sorted_list[12*j + 2] ,sorted_list[12*j +3] ,
#                                 sorted_list[12*j + 4] ,sorted_list[12*j + 5] ,sorted_list[12*j + 6] ,sorted_list[12*j + 7] ,
#                                 sorted_list[12*j + 8] ,sorted_list[12*j + 9] ,sorted_list[12*j + 10] ,sorted_list[12*j + 11] ,
#                                   sorted_list[12*j + 12] ,sorted_list[12*j + 13] ,sorted_list[12*j + 14] ,sorted_list[12*j +15] ,
#                                 sorted_list[12*j + 16] ,sorted_list[12*j + 17] ,sorted_list[12*j + 18] ,sorted_list[12*j + 19] ,
#                                 sorted_list[12*j + 20] ,sorted_list[12*j + 21] ,sorted_list[12*j + 22] ,sorted_list[12*j + 23]  ]
    
#     batch1['list_' + str(2*j + 1)] = [sorted_list[12*(j+1)] ,sorted_list[12*(j+1) + 1] ,sorted_list[12*(j+1) + 2] ,sorted_list[12*(j+1) +3] ,
#                                 sorted_list[12*(j+1) + 4] ,sorted_list[12*(j+1) + 5] ,sorted_list[12*(j+1) + 6] ,sorted_list[12*(j+1) + 7] ,
#                                 sorted_list[12*(j+1) + 8] ,sorted_list[12*(j+1) + 9] ,sorted_list[12*(j+1) + 10] ,sorted_list[12*(j+1) + 11] ,
#                                   sorted_list[12*(j+1) + 12] ,sorted_list[12*(j+1) + 13] ,sorted_list[12*(j+1) + 14] ,sorted_list[12*(j+1) +15] ,
#                                 sorted_list[12*(j+1) + 16] ,sorted_list[12*(j+1) + 17] ,sorted_list[12*(j+1) + 18] ,sorted_list[12*(j+1) + 19] ,
#                                 sorted_list[12*(j+1) + 20] ,sorted_list[12*(j+1) + 21] ,sorted_list[12*(j+1) + 22] ,sorted_list[12*(j+1) + 23]  ]

# for j in range(5):
#     batch2['list_' + str(2*j)] = [sorted_list[24*2*j] ,sorted_list[24*2*j + 2] ,sorted_list[24*2*j + 4] ,sorted_list[24*2*j +6] ,
#                                 sorted_list[24*2*j + 8] ,sorted_list[24*2*j + 10] ,sorted_list[24*2*j + 12] ,sorted_list[24*2*j + 14] ,
#                                 sorted_list[24*2*j + 16] ,sorted_list[24*2*j + 18] ,sorted_list[24*2*j + 20] ,sorted_list[24*2*j + 22] ,
#                                 sorted_list[24*2*j + 24] ,sorted_list[24*2*j + 26] ,sorted_list[24*2*j + 28] ,sorted_list[24*2*j + 30] ,
#                                 sorted_list[24*2*j + 32] ,sorted_list[24*2*j + 34] ,sorted_list[24*2*j + 36] ,sorted_list[24*2*j + 38] ,
#                                 sorted_list[24*2*j + 40] ,sorted_list[24*2*j + 42] ,sorted_list[24*2*j + 44] ,sorted_list[24*2*j + 46]  ]
    
#     batch2['list_' + str(2*j + 1)] =  [sorted_list[24*2*j + 1] ,sorted_list[24*2*j + 3] ,sorted_list[24*j + 5] ,sorted_list[24*2*j +7] ,
#                                 sorted_list[24*2*j + 9] ,sorted_list[24*2*j + 11] ,sorted_list[24*2*j + 13] ,sorted_list[24*2*j + 15] ,
#                                 sorted_list[24*2*j + 17] ,sorted_list[24*2*j + 19] ,sorted_list[24*2*j + 21] ,sorted_list[24*2*j + 23] ,
#                                 sorted_list[24*2*j + 25] ,sorted_list[24*2*j + 27] ,sorted_list[24*j + 29] ,sorted_list[24*2*j + 31] ,
#                                 sorted_list[24*2*j + 33] ,sorted_list[24*2*j + 35] ,sorted_list[24*2*j + 37] ,sorted_list[24*2*j + 39] ,
#                                 sorted_list[24*2*j + 41] ,sorted_list[24*2*j + 43] ,sorted_list[24*2*j + 45] ,sorted_list[24*2*j + 47]  ]
    
# for j in range(11):
    
#     batch1['list_' + str(j)] = [sorted_list[24*j] ,sorted_list[24*j + 1] ,sorted_list[24*j + 2] ,sorted_list[24*j +3] ,
#                                 sorted_list[24*j + 4] ,sorted_list[24*j + 5] ,sorted_list[24*j + 6] ,sorted_list[24*j + 7] ,
#                                 sorted_list[24*j + 8] ,sorted_list[24*j + 9] ,sorted_list[24*j + 10] ,sorted_list[24*j + 11] ,
#                                 sorted_list[24*j + 12] ,sorted_list[24*j + 13] ,sorted_list[24*j + 14] ,sorted_list[24*j +15] ,
#                                 sorted_list[24*j + 16] ,sorted_list[24*j + 17] ,sorted_list[24*j + 18] ,sorted_list[24*j + 19] ,
#                                 sorted_list[24*j + 20] ,sorted_list[24*j + 21] ,sorted_list[24*j + 22] ,sorted_list[24*j + 23]  ]


    
    # batch1['list_' + str(2*j + 1)] = [sorted_list[12*(j+1)] ,sorted_list[12*(j+1) + 1] ,sorted_list[12*(j+1) + 2] ,sorted_list[12*(j+1) +3] ,
    #                             sorted_list[12*(j+1) + 4] ,sorted_list[12*(j+1) + 5] ,sorted_list[12*(j+1) + 6] ,sorted_list[12*(j+1) + 7] ,
    #                             sorted_list[12*(j+1) + 8] ,sorted_list[12*(j+1) + 9] ,sorted_list[12*(j+1) + 10] ,sorted_list[12*(j+1) + 11] ,
    #                               sorted_list[12*(j+1) + 12] ,sorted_list[12*(j+1) + 13] ,sorted_list[12*(j+1) + 14] ,sorted_list[12*(j+1) +15] ,
    #                             sorted_list[12*(j+1) + 16] ,sorted_list[12*(j+1) + 17] ,sorted_list[12*(j+1) + 18] ,sorted_list[12*(j+1) + 19] ,
    #                             sorted_list[12*(j+1) + 20] ,sorted_list[12*(j+1) + 21] ,sorted_list[12*(j+1) + 22] ,sorted_list[12*(j+1) + 23]  ]


for j in range(5):
    batch1['list_' + str(2*j)] = [sorted_list[24*2*j] ,sorted_list[24*2*j + 2] ,sorted_list[24*2*j + 4] ,sorted_list[24*2*j +6] ,
                                sorted_list[24*2*j + 8] ,sorted_list[24*2*j + 10] ,sorted_list[24*2*j + 12] ,sorted_list[24*2*j + 14] ,
                                sorted_list[24*2*j + 16] ,sorted_list[24*2*j + 18] ,sorted_list[24*2*j + 20] ,sorted_list[24*2*j + 22] ,
                                sorted_list[24*2*j + 24] ,sorted_list[24*2*j + 26] ,sorted_list[24*2*j + 28] ,sorted_list[24*2*j + 30] ,
                                sorted_list[24*2*j + 32] ,sorted_list[24*2*j + 34] ,sorted_list[24*2*j + 36] ,sorted_list[24*2*j + 38] ,
                                sorted_list[24*2*j + 40] ,sorted_list[24*2*j + 42] ,sorted_list[24*2*j + 44] ,sorted_list[24*2*j + 46]  ]
    
    batch1['list_' + str(2*j + 1)] =  [sorted_list[24*2*j + 1] ,sorted_list[24*2*j + 3] ,sorted_list[24*j + 5] ,sorted_list[24*2*j +7] ,
                                sorted_list[24*2*j + 9] ,sorted_list[24*2*j + 11] ,sorted_list[24*2*j + 13] ,sorted_list[24*2*j + 15] ,
                                sorted_list[24*2*j + 17] ,sorted_list[24*2*j + 19] ,sorted_list[24*2*j + 21] ,sorted_list[24*2*j + 23] ,
                                sorted_list[24*2*j + 25] ,sorted_list[24*2*j + 27] ,sorted_list[24*j + 29] ,sorted_list[24*2*j + 31] ,
                                sorted_list[24*2*j + 33] ,sorted_list[24*2*j + 35] ,sorted_list[24*2*j + 37] ,sorted_list[24*2*j + 39] ,
                                sorted_list[24*2*j + 41] ,sorted_list[24*2*j + 43] ,sorted_list[24*2*j + 45] ,sorted_list[24*2*j + 47]  ]
    
for j in range(10):
    
    batch0['list_' + str(j)] = [sorted_list[24*j] ,sorted_list[24*j + 1] ,sorted_list[24*j + 2] ,sorted_list[24*j +3] ,
                                sorted_list[24*j + 4] ,sorted_list[24*j + 5] ,sorted_list[24*j + 6] ,sorted_list[24*j + 7] ,
                                sorted_list[24*j + 8] ,sorted_list[24*j + 9] ,sorted_list[24*j + 10] ,sorted_list[24*j + 11] ,
                                sorted_list[24*j + 12] ,sorted_list[24*j + 13] ,sorted_list[24*j + 14] ,sorted_list[24*j +15] ,
                                sorted_list[24*j + 16] ,sorted_list[24*j + 17] ,sorted_list[24*j + 18] ,sorted_list[24*j + 19] ,
                                sorted_list[24*j + 20] ,sorted_list[24*j + 21] ,sorted_list[24*j + 22] ,sorted_list[24*j + 23]  ]

j = 10
batch1['list_' + str(j)] = [sorted_list[12*2*j] ,sorted_list[12*2*j + 2] ,sorted_list[12*2*j + 4] ,sorted_list[12*2*j +6] ,
                            sorted_list[12*2*j + 8] ,sorted_list[12*2*j + 10] ,sorted_list[12*2*j + 12] ,sorted_list[12*2*j + 14] ,
                            sorted_list[12*2*j + 16] ,sorted_list[12*2*j + 18] ,sorted_list[12*2*j + 20] ,sorted_list[12*2*j + 22]  ]
    
batch1['list_' + str(j + 1)] =  [sorted_list[12*2*j + 1] ,sorted_list[12*2*j + 3] ,sorted_list[12*j + 5] ,sorted_list[12*2*j +7] ,
                            sorted_list[12*2*j + 9] ,sorted_list[12*2*j + 11] ,sorted_list[12*2*j + 13] ,sorted_list[12*2*j + 15] ,
                            sorted_list[12*2*j + 17] ,sorted_list[12*2*j + 19] ,sorted_list[12*2*j + 21] ,sorted_list[12*2*j + 23]  ]

batch0['list_' + str(j)] = [sorted_list[12*2*j] ,sorted_list[12*2*j + 1] ,sorted_list[12*2*j + 2] ,sorted_list[12*2*j +3] ,
                            sorted_list[12*2*j + 4] ,sorted_list[12*2*j + 5] ,sorted_list[12*2*j + 6] ,sorted_list[12*2*j + 7] ,
                            sorted_list[12*2*j + 8] ,sorted_list[12*2*j + 9] ,sorted_list[12*2*j + 10] ,sorted_list[12*2*j + 11]  ]

batch0['list_' + str(j + 1)] = [sorted_list[12*2*j + 12] ,sorted_list[12*2*j + 12 + 1] ,sorted_list[12*2*j + 12 + 2] ,sorted_list[12*2*j + 12 +3] ,
                            sorted_list[12*2*j + 12 + 4] ,sorted_list[12*2*j + 12 + 5] ,sorted_list[12*2*j + 12 + 6] ,sorted_list[12*2*j + 12 + 7] ,
                            sorted_list[12*2*j + 12 + 8] ,sorted_list[12*2*j + 12 + 9] ,sorted_list[12*2*j + 12 + 10] ,sorted_list[12*2*j + 12 + 11]  ]

    

for quelbatch in range(2):
    for n_list in range(12):

        if n_list == 0 or n_list == 1:
        # if True:

            # quelbatch = 1
            # n_list = 21
            # special_rating = 0

            # n_list = 21
            # if quelbatch == 0 and n_list == 2:
            #     flush_every = 50
            # else:
            #     flush_every = 150
            test_ind = []
            used_ind = []

            # if (quelbatch == 0 and (n_list < 9)):
            if False:
                print('avoided')
            else:
                print('batch : ',quelbatch,' n_list : ',n_list)
                with open('infos.txt', 'a') as f:
                    f.write('batch : ' + str(quelbatch) +  ' n_list : ' +  str(n_list))
                    f.write('\n')
                

                os.makedirs('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/train/')
                os.makedirs('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/test/')
                os.makedirs('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/calib/')
                # os.makedirs('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/train/')
                # os.makedirs('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/test/')
                # os.makedirs('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/calib/')

                
                if quelbatch  == 0:
                    batch = batch0
                else: 
                    batch = batch1
                    
                training_list = batch['list_' + str(n_list)]
                species_dict = {}
                for i,species in enumerate(training_list):
                    species_dict[species] = i
                
                train_metadata = pd.read_csv('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/train_metadata.csv')

                train_metadata.insert(column='selected',loc=1,value=int)
                for x in train_metadata.itertuples():
                    train_metadata.at[x.Index,'selected'] = select_training_list(x.primary_label,species_dict,x.rating,n_list,quelbatch)

                train = train_metadata[train_metadata['selected'] == 1].sample(frac=0.99,random_state=0)
                # test_calib = train_metadata[train_metadata['selected'] == 1].drop(train.index)
                # test = test_calib.sample(frac=0.5,random_state=0)
                # calib = test_calib.drop(test.index)

                train_metadata.insert(column='ttc',loc=2,value=str)
                for x in train_metadata.itertuples():
                    if x.Index in train.index:
                        train_metadata.at[x.Index,'tort'] = 'train'
                    # elif x.Index in test.index:
                    #     train_metadata.at[x.Index,'tort'] = 'test'
                    # elif x.Index in calib.index:
                    #     train_metadata.at[x.Index,'tort'] = 'calib'

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
                    # print(samplerate)
                    data = np.asarray(data)
                    # data = transform(data)
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
                            t = data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack]
                            y = librosa.feature.mfcc(y = t, sr = samplerate,n_mfcc = 126)
                            t = torch.from_numpy(y)
                            torch.save(t, 'data/subtracks64000/train/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
                            k += 1
                            processed += 1
                    # elif x.tort == 'test':
                    # elif tort == 'test':
                    #     for i in range(track_length // steps_per_subtrack):
                    #         # with torch.no_grad():
                    #             # input_values = processor(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], return_tensors="pt",sampling_rate = 16000).input_values.to('cuda:0')  # Batch size 1
                    #             # hidden_states = model(input_values.squeeze(0)).last_hidden_state
                    #             # subtrack_index = str(k)
                    #             # torch.save(hidden_states, 'data/subtracks64000/test/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
                    #             # k += 1
                    #         subtrack_index = str(k)
                    #         t = data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack]
                    #         y = librosa.feature.mfcc(y = t, sr = samplerate,n_mfcc = 126)
                    #         t = torch.from_numpy(y)
                    #         torch.save(t, 'data/subtracks64000/test/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
                    #         k += 1
                    #         processed += 1
                    #     test_ind.append(ind)
                    # # elif x.tort == 'calib':
                    # elif tort == 'calib':
                    #     for i in range(track_length // steps_per_subtrack):
                    #         # with torch.no_grad():
                    #             # input_values = processor(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], return_tensors="pt",sampling_rate = 16000).input_values.to('cuda:0')  # Batch size 1
                    #             # hidden_states = model(input_values.squeeze(0)).last_hidden_state
                    #             # subtrack_index = str(k)
                    #             # torch.save(hidden_states, 'data/subtracks64000/calib/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
                    #             # k += 1
                    #         subtrack_index = str(k)
                    #         t = data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack]
                    #         y = librosa.feature.mfcc(y = t, sr = samplerate,n_mfcc = 126)
                    #         t = torch.from_numpy(y)
                    #         torch.save(t, 'data/subtracks64000/calib/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
                    #         k += 1
                    #         processed += 1
                    
                    
                    
                    # if (processed > flush_every ):
                    #     processed_all += processed
                    #     processed = 0
                        # print(processed_all)
                        # end = time.time()
                        # print('x')
                        # print(end - start)
                        # start = time.time()
                        # wav2vec_processing('train/')
                        # print('x_traon')
                        # end = time.time()
                        # print(end - start)
                        # start = time.time()
                        # wav2vec_processing('test/')
                        # print('x_test')
                        # end = time.time()
                        # print(end - start)
                        # wav2vec_processing('calib/') 

                    if (processed > limit):
                        limit_reached = True
                        break
                print('processed all')    
                print(processed)

                # if processed_all + processed < limit :
                #     wav2vec_processing('train/')
                #     wav2vec_processing('test/')
                #     wav2vec_processing('calib/')
                #     shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/train/')
                #     shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/test/')
                #     shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/calib/')
                # else:
                #     shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/train/')
                #     shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/test/')
                #     shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/calib/')

                ### export information
                ### train
                end = time.time()
                print('process ended : ')
                print(end-start)
                model_class = trainer(quelbatch ,n_list)

                
                # with open('infos.txt', 'a') as f:
                #     f.write('test indices')
                #     f.write('\n')
                #     q = ' '.join(str(e) for e in test_ind)
                #     f.write(q)
                #     f.write('\n')
                #     f.write('used indices')
                #     f.write('\n')
                #     q = ' '.join(str(e) for e in used_ind)
                #     f.write(q)
                #     f.write('\n')
                # ### size reduction for real calibration
                # if n_list > 9 :
                #     model_classifier = cinnv2()
                # else:
                #      model_classifier = cinnv3()
                # saving_path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/batch' + str(quelbatch) + '/nlist_' + str(n_list)
                # model_classifier.load_state_dict(torch.load(saving_path,map_location='cpu'))
                # model = copy.deepcopy(model_classifier)
                # model.eval()
                # # `qconfig` means quantization configuration, it specifies how should we
                # # observe the activation and weight of an operator
                # # `qconfig_dict`, specifies the `qconfig` for each operator in the model
                # # we can specify `qconfig` for certain types of modules
                # # we can specify `qconfig` for a specific submodule in the model
                # # we can specify `qconfig` for some functioanl calls in the model
                # # we can also set `qconfig` to None to skip quantization for some operators
                # # qconfig = get_default_qconfig_mapping("fbgemm") get_default_qconfig
                # qconfig = get_default_qconfig("fbgemm")
                # qconfig_dict = {"": qconfig}
                # # qconfig_dict = qconfig.to_dict()
                # # `prepare_fx` inserts observers in the model based on the configuration in `qconfig_dict`
                # model_prepared = prepare_fx(model, qconfig_dict,torch.randn(1, 1, 126, 126))
                # # calibration runs the model with some sample data, which allows observers to record the statistics of
                # # the activation and weigths of the operators
                # calibration_data = [torch.randn(1, 1, 126, 126) for _ in range(100)]
                # for i in range(len(calibration_data)):
                #     model_prepared(calibration_data[i])
                # # `convert_fx` converts a calibrated model to a quantized model, this includes inserting
                # # quantize, dequantize operators to the model and swap floating point operators with quantized operators
                # model_quantized = convert_fx(copy.deepcopy(model_prepared))
                # ### calibrate
                # if n_list < 10:
                #     calibratev3(model_quantized,species_dict,[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8])
                # else:
                #     calibratev2(model_quantized,species_dict,[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8])
                shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/train/')
                shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/test/')
                shutil.rmtree('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/calib/')
                ### export information
