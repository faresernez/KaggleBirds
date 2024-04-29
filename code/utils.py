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
# from calibration2 import calibratev3
# from calibration import calibratev2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import matplotlib.pyplot as plt

def audio_to_chunks(audio_file,steps_per_subtrack = 160000, sr=32000):
    chunks = []
    data, samplerate = librosa.load(audio_file, sr=sr)
    # print(samplerate)
    # print(type(data))
    # print(data.shape)
    track_length = data.shape[0]
    for i in range(track_length // steps_per_subtrack):
        chunks.append(data[i*steps_per_subtrack:(i+1)*steps_per_subtrack])
    # print(len(chunks))
    return chunks , samplerate

def chunk_to_spectrum(chunk,samplerate,n_mels=224, hop_length=512, fmax=16000): # S.shape = (X,224,313)
    # S = []
    # for chunk in chunks:
    S = librosa.feature.melspectrogram(y=chunk, sr=samplerate, n_mels=n_mels, hop_length=hop_length, fmax=fmax)
    # mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S))
    # print('shape of mfccs : ', S.shape) #(224,313)
    return S

class CustomDataLoader():

    def __init__(self,batchSize,dataPath):
        self.batchSize = batchSize
        self.dataPath = dataPath
        self.filesMap = {}
        for i,path in enumerate(Path(dataPath).glob('*')):
            self.filesMap[i] = path
        self.nFiles = i+1
        self.index = range(self.nFiles)
        self.posResumeInIndex = 0
        self.posResumeInFile = 0

    def startEpoch(self):
        random.shuffle(self.index)
        self.posResumeInIndex = 0
        self.posResumeInFile = 0


    def load_batch(self):
        nDataPoints = 0
        batchUncomplete = True
        batch = []
        while (batchUncomplete and self.posResumeInIndex < self.nFiles):
            chunks, samplerate = audio_to_chunks(self.filesMap[self.index[self.posResumeInIndex]])
            nChunks = len(chunks)
            if ((nChunks - self.posResumeInFile) <= (self.batchSize - nDataPoints)):
                for chunk in chunks[self.posResumeInFile:]:
                    batch.append(chunk_to_spectrum(chunk,samplerate))
                self.posResumeInIndex += 1
                self.posResumeInFile = 0
                nDataPoints = len(batch)
            else:
                newPos = self.posResumeInFile + (self.batchSize - nDataPoints)
                for chunk in chunks[self.posResumeInFile:newPos]:
                    batch.append(chunk_to_spectrum(chunk,samplerate))
                self.posResumeInFile = newPos
                nDataPoints = len(batch)
            if (nDataPoints == self.batchSize):
                batchUncomplete = False
        return batch
    
# DataLoader = CustomDataLoader(24,r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\Birdclef2024\unlabeled_soundscapes')
# print(DataLoader.posResumeInIndex)
# print(DataLoader.posResumeInFile)
# for j in range(10):
#     print('batch number : ', j)
#     b = DataLoader.load_batch()
#     print(DataLoader.posResumeInIndex)
#     print(DataLoader.posResumeInFile)

# DataLoader = CustomDataLoader(48,r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\Birdclef2024\unlabeled_soundscapes')
# print(DataLoader.posResumeInIndex)
# print(DataLoader.posResumeInFile)
# for j in range(5):
#     print('batch number : ', j)
#     b = DataLoader.load_batch()
#     print(DataLoader.posResumeInIndex)
#     print(DataLoader.posResumeInFile)

                










# audio_file = r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\Birdclef2024\train_audio\asbfly\XC49755.ogg'
# chunks , samplerate = audio_to_chunks(audio_file)
# print(len(chunks))
# # print(chunks[0].shape)
# S = chunk_to_spectrum(chunks[0],samplerate)

