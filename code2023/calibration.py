
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
import math

soft = nn.Softmax(dim=1)

class LSTM(nn.Module):
    """
    input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediciton_periods input to get_x_y_pairs
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers = 1)
        
        self.linear = nn.Linear(hidden_size, output_size)

        # self.linear2 = nn.Linear(output_size, 12)

        # self.relu = nn.ReLU()
        
    def forward(self, x, hidden=None):
        if hidden==None:
            self.hidden = (torch.zeros(1,1,self.hidden_size).to('cuda:0'),
                           torch.zeros(1,1,self.hidden_size).to('cuda:0'))
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
        # predictions = self.relu(predictions)
        # predictions = self.linear2(predictions)
        # print('pred')
        # print(predictions.shape)
        # print('-1')
        # print(predictions[-1].shape)
        
        return predictions[-1], self.hidden

def pathToSpecies(path,dic):
    return dic[path.split('_')[2]]

def calibratev2(model : nn.Module, species_dict, alphaList , delta = 0.1, tcv = False):


    soft = nn.Softmax(dim=1)
    # alphaList = [0.1,0.05]
    # # alphaList = [0.1,0.05,0.01]
    # # alphaList = [0.1]
    # delta = 0.1
    # R = 100
    # tcv = True
    # label_conditional = False
    model.eval()

    path_calib = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/calib/'
    l_calib = os.listdir(path_calib)
    # print(l)
    q_hat_dic = {}

    batch_size = 4
    nelem = len(l_calib)
    n_loops = nelem // batch_size
    reste = nelem % batch_size

    if tcv == True:
        f = 2
    else:
        f = 1
    
    with torch.no_grad():
        for j in range(n_loops):
            t = torch.load(path_calib + l_calib[j*batch_size], map_location='cpu').unsqueeze(dim=0)
            y = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_calib[j*batch_size],species_dict)
            for i in range(batch_size - 1):
                    x = torch.load(path_calib + l_calib[j*batch_size + i + 1], map_location='cpu').unsqueeze(dim=0)
                    z = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_calib[j*batch_size + i + 1],species_dict)
                    t = torch.cat((t,x),dim=0)
                    # print('b')
                    # print(t.shape)
                    y = torch.cat((y,z),dim=0)
            y_hat = model(t)
            if j == 0:
                ypred_calib = y_hat
                yt_calib = y
            else:
                ypred_calib = torch.cat((ypred_calib,y_hat),dim=0)
                yt_calib = torch.cat((yt_calib,y),dim=0)
        if reste != 0:
            done = batch_size * n_loops
            t = torch.load(path_calib + l_calib[done], map_location='cpu').unsqueeze(dim=0)
            # y = torch.as_tensor(pathToSpecies(l_train[done],species_dict)).to('cuda:0')
            y = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_calib[done],species_dict)
            for i in range(reste - 1):
                x = torch.load(path_calib + l_calib[done + i + 1], map_location='cpu').unsqueeze(dim=0)
                z = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_calib[done + i + 1],species_dict)
                t = torch.cat((t,x),dim=0)
                y = torch.cat((y,z),dim=0)
            y_hat = model(t)
            ypred_calib = torch.cat((ypred_calib,y_hat),dim=0)
            yt_calib = torch.cat((yt_calib,y),dim=0)
            ypred_calib = ypred_calib.numpy()
            yt_calib = yt_calib.numpy()



    # with torch.no_grad():
    #     t = torch.load(path_calib + l_calib[0], map_location='cuda:0').unsqueeze(dim=0)
    #     ypred_calib , _ = model(t,1,None)
    #     # print('1')
    #     # print(ypred_calib.shape)
    #     # print(ypred_calib)
    #     ypred_calib  = soft(ypred_calib)
    #     # print('2')
    #     # print(ypred_calib.shape)
    #     # print(ypred_calib)
    # yt_calib = [pathToSpecies(l_calib[0],species_dict)]
    # for file in l_calib[1:]:
    #     with torch.no_grad():
    #         t = torch.load(path_calib + file, map_location='cuda:0').unsqueeze(dim=0)
    #         x , _ = model(t,1,None)
    #         # print(x)
    #         # print(x.shape)
    #         # x  = soft(x.unsqueeze(dim=0))
    #         x  = soft(x)
    #         # print(x)
    #         # print(x.shape)
    #         ypred_calib = torch.cat((ypred_calib,x),dim=0)
    #     yt_calib.append(pathToSpecies(file,species_dict))
    
    # print('ypredcalib')
    # print(ypred_calib.shape)
    # print(len(yt_calib))
    # print(yt_calib)

    for l in range(f): #in range(2) if with or without tcv


        words2CalibList = []
        words0CalibList = []
        words1CalibList = []
        words2TestList = []
        words0TestList = []
        words1TestList = []


        meanSetSizeList = []
        coverageMarginalList = []
        coverage0lList = []
        coverage1lList = []
        ratio0List = []
        ratio1List = []
        ratio2List = []


        conf_pred01 = []
        conf_pred005 = []
        conf_pred001 = []
        len_conf_pred01 = []
        len_conf_pred005 = []
        len_conf_pred001 = []


        if l == 0:
            tcv = True
        else:
            tcv = False
            
        for alpha in alphaList:
            # print(alpha)
            # lam = []
            lam_labels = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],
                          7:[],8:[],9:[],10:[],11:[],12:[]}
            q_hat = {0:0.,1:0.,2:0.,3:0.,4:0.,5:0.,6:0.,
                          7:0.,8:0.,9:0.,10:0.,11:0.,12:0.}
            q_hat_tcv = {0:0.,1:0.,2:0.,3:0.,4:0.,5:0.,6:0.,
                          7:0.,8:0.,9:0.,10:0.,11:0.,12:0.}
            n_calib = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,
                          7:0,8:0,9:0,10:0,11:0,12:0}
            # print('zeb')
            # print(ypred_calib)
            for i in range(ypred_calib.shape[0]):
                # print('az')
                # print(ypred_calib.shape)
                # print(ypred_calib.cpu().numpy().shape)
                # print(ypred_calib.cpu().numpy())
                lam_labels[12].append(1 - ypred_calib[i,int(yt_calib[i])])
                lam_labels[int(yt_calib[i])].append(1 - ypred_calib[i,int(yt_calib[i])])

            for key in list(lam_labels.keys()):

                n_calib[key] = len(lam_labels[key])
                # print(key)
                # print(n_calib[key])
                if n_calib[key] == 0:
                    alpha_modif = -1
                elif (tcv):
                    alpha_modif = alpha - math.sqrt(-math.log(delta)/(2*n_calib[key]))
                    # print(alpha_modif)
                else:
                    alpha_modif = alpha


                if alpha_modif < 0 :
                    # print('Conditional-Training ICP is impossible with alpha = ', )
                    # df_words = df_words.drop(columns='type')
                    q_hat[key] = np.NaN
                else:
                    q_hat[key] = np.quantile(lam_labels[key],np.floor((n_calib[key]+1)*(1-alpha_modif))/n_calib[key],method='nearest')

            # print(q_hat)
            q_hat_dic[alpha] = q_hat
    
    q_hat_string = ''
    for key,value in q_hat_dic.items():
        s = str(key) + ' : ' + str(value)
        q_hat_string += s
    with open('infos.txt', 'a') as f:
        f.write('q_hat')
        f.write('\n')
        f.write(q_hat_string)
        f.write('\n')
    
    del ypred_calib
    del yt_calib
    del lam_labels
    del q_hat
    del n_calib

    path_test = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/test/'
    l_test = os.listdir(path_test)
    # print(l)

    if tcv == True:
        f = 2
    else:
        f = 1
    
    # with torch.no_grad():
    #     t = torch.load(path_test + l_test[0], map_location='cuda:0').unsqueeze(dim=0)
    #     ypred_test , _ = model(t,1,None)
    #     # print('1')
    #     # print(ypred_test.shape)
    #     # print(ypred_test)
    #     ypred_test  = soft(ypred_test)
    #     # print('2')
    #     # print(ypred_test.shape)
    #     # print(ypred_test)
    # yt_test = [pathToSpecies(l_test[0],species_dict)]
    # for file in l_test[1:]:
    #     with torch.no_grad():
    #         t = torch.load(path_test + file, map_location='cuda:0').unsqueeze(dim=0)
    #         x , _ = model(t,1,None)
    #         x  = soft(x)
    #         ypred_test = torch.cat((ypred_test,x),dim=0)
    #     yt_test.append(pathToSpecies(file,species_dict)) 
    
    # ypred_test.cpu().numpy()

    nelem = len(l_test)
    n_loops = nelem // batch_size
    reste = nelem % batch_size

    with torch.no_grad():
        for j in range(n_loops):
            t = torch.load(path_test + l_test[j*batch_size], map_location='cpu').unsqueeze(dim=0)
            y = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_test[j*batch_size],species_dict)
            for i in range(batch_size - 1):
                    x = torch.load(path_test + l_test[j*batch_size + i + 1], map_location='cpu').unsqueeze(dim=0)
                    z = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_test[j*batch_size + i + 1],species_dict)
                    t = torch.cat((t,x),dim=0)
                    # print('b')
                    # print(t.shape)
                    y = torch.cat((y,z),dim=0)
            y_hat = model(t)
            if j == 0:
                ypred_test = y_hat
                yt_test = y
            else:
                ypred_test = torch.cat((ypred_test,y_hat),dim=0)
                yt_test = torch.cat((yt_test,y),dim=0)
        if reste != 0:
            done = batch_size * n_loops
            t = torch.load(path_test + l_test[done], map_location='cpu').unsqueeze(dim=0)
            # y = torch.as_tensor(pathToSpecies(l_train[done],species_dict)).to('cpu')
            y = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_test[done],species_dict)
            for i in range(reste - 1):
                x = torch.load(path_test + l_test[done + i + 1], map_location='cpu').unsqueeze(dim=0)
                z = torch.ones(1,device='cpu',dtype=torch.long)*pathToSpecies(l_test[done + i + 1],species_dict)
                t = torch.cat((t,x),dim=0)
                y = torch.cat((y,z),dim=0)
            y_hat = model(t)
            ypred_test = torch.cat((ypred_test,y_hat),dim=0)
            yt_test = torch.cat((yt_test,y),dim=0)
            ypred_test = ypred_test.numpy()
            yt_test = yt_test.numpy()

    for alpha in alphaList:
        with open('infos.txt', 'a') as f:
            f.write('alpha : ' +  str(alpha))
            f.write('\n')
        tcv_size = []
        tcv_coverage = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],
                          7:[],8:[],9:[],10:[],11:[],12:[]}
        coverage = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],
                          7:[],8:[],9:[],10:[],11:[],12:[]}
        notcv_size = []
        notcv_coverage = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],
                          7:[],8:[],9:[],10:[],11:[],12:[]}
        
        tcv_label_conditional_possible = True
        tcv_possible = True

        # print(q_hat_dic[alpha])
        for key,value in q_hat_dic[alpha].items():
            # print(key)
            # print(value)
            if key == 12 and np.isnan(value):
                tcv_possible = False
                tcv_label_conditional_possible =  False
            elif key != 12 and np.isnan(value):
                tcv_label_conditional_possible = False

        # print(tcv_possible)
        # print(tcv_label_conditional_possible)

        if tcv_label_conditional_possible and tcv_possible:
            for i,elem in enumerate(ypred_test):
                pred_tcv = []
                pred_notcv = []
                # elem = ypred_calib[i]
                for classe,soft in enumerate(elem):
                    if (1 - soft) <= q_hat_dic[alpha][12]:
                        pred_notcv.append(classe)
                    if (1 - soft) <= q_hat_dic[alpha][classe]:
                        pred_tcv.append(classe)
                        # if classe == 0 and (1 - soft) <= q_hat_0:
                        #     pred.append(classe)
                        # elif classe == 1 and (1 - soft) <= q_hat_1:
                        #     pred.append(classe)
                if pred_notcv == []:
                    # print('nothing was predicted')
                    pred_notcv = [0.,1.,2.,3.,4.,5.,
                                    6.,7.,8.,9.,10.,11.]
                if pred_tcv == []:
                    # print('nothing was predicted')
                    pred_tcv = [0.,1.,2.,3.,4.,5.,
                                    6.,7.,8.,9.,10.,11.]
                
                tcv_size.append(len(pred_tcv))
                notcv_size.append(len(pred_notcv))
                
                true_label = yt_test[i]
                
                if true_label in pred_notcv:
                    coverage[12].append(1.)
                else:
                    coverage[12].append(0.)
                if true_label in pred_tcv:
                    coverage[true_label].append(1.)
                else:
                    coverage[true_label].append(0.)
            
            for key in list(coverage.keys()):
                coverage[key] = np.mean(coverage[key])
            
            with open('infos.txt', 'a') as f:
                # f.write('alpha : ',alpha)
                # f.write('\n')
                f.write('label_conditional_set_size : ' +  str(np.mean(tcv_size)))
                f.write('\n')
                f.write('no_label_conditional_set_size : ' + str(np.mean(notcv_size)))
                f.write('\n')
                f.write('coverage')
                f.write('\n')
                f.write(str(coverage))
                f.write('\n')

        elif tcv_possible:
            # print('label conditional not possible')
            for i,elem in enumerate(ypred_test):
                pred_notcv = []
                # elem = ypred_calib[i]
                for classe,soft in enumerate(elem):
                    if (1 - soft) <= q_hat_dic[alpha][12]:
                        pred_notcv.append(classe)
                if pred_notcv == []:
                    # print('nothing was predicted')
                    pred_notcv = [0.,1.,2.,3.,4.,5.,
                                    6.,7.,8.,9.,10.,11.]
                
                notcv_size.append(len(pred_notcv))
                
                true_label = yt_test[i]
                
                if true_label in pred_notcv:
                    coverage[12].append(1.)
                else:
                    coverage[12].append(0.)
            
            for key in list(coverage.keys()):
                if key == 12:
                    coverage[key] = np.mean(coverage[key])
            
            # print('tcv_set_size : ', np.mean(tcv_size))
            # print('notcv_set_size : ', np.mean(notcv_size))
            # print('coverage : ', coverage[12])
            # print(coverage[12])
            with open('infos.txt', 'a') as f:
                # f.write('alpha : ',alpha)
                # f.write('\n')
                # f.write('label_conditional_set_size : ', np.mean(tcv_size))
                # f.write('\n')
                f.write('no_label_conditional_set_size : ' +  str(np.mean(notcv_size)))
                f.write('\n')
                f.write('coverage')
                f.write('\n')
                f.write(str(coverage[12]))
                f.write('\n')
        else:
            print('training conditional impossible')
            with open('infos.txt', 'a') as f:
                f.write('training conditional impossible')
                f.write('\n')



# def ratingfn(n,batch):
#     if n < 7 and batch == 0:
#         return 3
#     elif n < 3 and batch == 1:
#         return 3
#     else:
#         return 0
    
# def select_training_list(species,training_list,rating,n_list,batch):
#             return species in training_list and rating > ratingfn(n_list,batch)


# steps_per_subtrack = 32000
# flush_every = 200
# limit = 14000

# transform = torchaudio.transforms.Resample(32000, 16000)
# path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/train_audio/'

# train_metadata = pd.read_csv('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/train_metadata.csv')

# general_species_dict = {}
# size_dict = {}
# pathlist = Path(r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\train_audio').glob('*')
# i = 0
# for path in pathlist:
#     path_in_str = str(path)
#     species = path_in_str[60:]
#     general_species_dict[species] = i
#     s = 0
#     for ele in os.scandir(path_in_str):
#         s += os.path.getsize(ele)
#     size_dict[species] = s
#     i += 1
# sorted_size_dict = sorted(size_dict.items(), key=lambda x:x[1], reverse=True)
# converted_dict = dict(sorted_size_dict)

# sorted_list = list(converted_dict.keys())

# batch1 = {}
# batch2 = {}
# for j in range(11):
#     batch2['list_' + str(2*j)] = [sorted_list[12*2*j] ,sorted_list[12*2*j + 2] ,sorted_list[12*2*j + 4] ,sorted_list[12*2*j +6] ,
#                                 sorted_list[12*2*j + 8] ,sorted_list[12*2*j + 10] ,sorted_list[12*2*j + 12] ,sorted_list[12*2*j + 14] ,
#                                 sorted_list[12*2*j + 16] ,sorted_list[12*2*j + 18] ,sorted_list[12*2*j + 20] ,sorted_list[12*2*j + 22]  ]
    
#     batch2['list_' + str(2*j + 1)] =  [sorted_list[12*2*j + 1] ,sorted_list[12*2*j + 3] ,sorted_list[12*j + 5] ,sorted_list[12*2*j +7] ,
#                                 sorted_list[12*2*j + 9] ,sorted_list[12*2*j + 11] ,sorted_list[12*2*j + 13] ,sorted_list[12*2*j + 15] ,
#                                 sorted_list[12*2*j + 17] ,sorted_list[12*2*j + 19] ,sorted_list[12*2*j + 21] ,sorted_list[12*2*j + 23]  ]
    
#     batch1['list_' + str(2*j)] = [sorted_list[12*j] ,sorted_list[12*j + 1] ,sorted_list[12*j + 2] ,sorted_list[12*j +3] ,
#                                 sorted_list[12*j + 4] ,sorted_list[12*j + 5] ,sorted_list[12*j + 6] ,sorted_list[12*j + 7] ,
#                                 sorted_list[12*j + 8] ,sorted_list[12*j + 9] ,sorted_list[12*j + 10] ,sorted_list[12*j + 11]  ]
    
#     batch1['list_' + str(2*j + 1)] = [sorted_list[12*(j+1)] ,sorted_list[12*(j+1) + 1] ,sorted_list[12*(j+1) + 2] ,sorted_list[12*(j+1) +3] ,
#                                 sorted_list[12*(j+1) + 4] ,sorted_list[12*(j+1) + 5] ,sorted_list[12*(j+1) + 6] ,sorted_list[12*(j+1) + 7] ,
#                                 sorted_list[12*(j+1) + 8] ,sorted_list[12*(j+1) + 9] ,sorted_list[12*(j+1) + 10] ,sorted_list[12*(j+1) + 11]  ]


# quelbatch = 0
# n_list = 0
# # special_rating = 0

# if quelbatch  + 1 == 1:
#     batch = batch1
# else: 
#     batch = batch2
    
# training_list = batch['list_' + str(n_list)]
# species_dict = {}
# for i,species in enumerate(training_list):
#     species_dict[species] = i
# print(species_dict)

# # model = torch.load(r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\models\batch1\nlist_0')

# # model = TheModelClass(*args, **kwargs)
# model = LSTM(input_size=768, hidden_size=256, output_size=12).to('cuda:0')
# model.load_state_dict(torch.load(r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\models\batch1\nlist_0'))
# model.eval()

# calibrate(model,species_dict,[0.2,0.5,0.6,0.7,0.00000001])

# def testCalibration(model : nn.Module, species_dict, alphaList , delta = 0.1, tcv = False):
    
        
