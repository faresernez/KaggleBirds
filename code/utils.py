import shutil
import os
import librosa
import numpy as np
from pathlib import Path
import random
import torch


def audio_to_chunks(audio_file,steps_per_subtrack = 160000, sr=32000):
    chunks = []
    data, samplerate = librosa.load(audio_file, sr=sr)
    track_length = data.shape[0]
    nChunks = track_length // steps_per_subtrack
    if (nChunks == 0): #if an audio is shorter than steps_per_subtrack, we duplicate it
        while (data.shape[0] < steps_per_subtrack):
            data = np.tile(data,2)
        nChunks = 1
    for i in range(nChunks):
        chunks.append(data[i*steps_per_subtrack:(i+1)*steps_per_subtrack])
    chunks.append(data[-steps_per_subtrack:]) #adding the last steps_per_subtrack of the audio to not miss anything
    return chunks , samplerate

def chunk_to_spectrum(chunk,samplerate,n_mels=224, hop_length=716, fmax=16000): # S.shape = (X,224,313)
    S = librosa.feature.melspectrogram(y=chunk, sr=samplerate, n_mels=n_mels, hop_length=hop_length, fmax=fmax)
    S = librosa.util.normalize(S)
    # mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S))
    return S

def trainTestCalib(percentages = [0.8,0.2,0.]):
        r = random.random()
        if (percentages[0] == 1):
            return 'train/'
        elif (percentages[2] == 0 and r > percentages[1]):
            return 'train/'
        elif (percentages[2] == 0):
            return 'test/'
        elif (r < percentages[1]):
            return 1
        elif (r > 1 - percentages[2]):
            return 'calib/'
        else:
            return 'train/'

def extract(dataProcessor,ratioTrainTestCalib,dataPath,destination,classes,extractionDone):
        
        if (not extractionDone):
        
            folders = ['C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/Birdclef2024/finetuning/train/',
                    'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/Birdclef2024/finetuning/test/',
                    'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/Birdclef2024/finetuning/calib/']
            for folder in folders:
                shutil.rmtree(folder)
                os.mkdir(folder)

            BirdClassMap = {}
            filesMap = {}
            classMap = {}
            ind = 0 
            indDict = {'train/':[],'test/':[],'calib/':[]}
            for i,c in enumerate(classes):
                stri = str(i)
                BirdClassMap[c] = i
                classPath = dataPath + c + '/'
                for path in Path(classPath).glob('*'):
                    chunks = dataProcessor.loadAudio(path)
                    for chunk in chunks:
                        folder = trainTestCalib(ratioTrainTestCalib)
                        tensorPath = destination + folder + str(ind) + '_' + stri + '.pt'
                        torch.save(torch.from_numpy(dataProcessor.processChunk(chunk)),tensorPath)
                        indDict[folder].append(ind)
                        filesMap[ind] = tensorPath
                        classMap[ind] = i
                        ind += 1
            return BirdClassMap , indDict['train/'] , indDict['test/'] , indDict['calib/'] , filesMap , classMap
        
        else:

            BirdClassMap = {}
            filesMap = {}
            classMap = {}
            ind = 0 
            indDict = {'train/':[],'test/':[],'calib/':[]}
            for i,c in enumerate(classes):
                stri = str(i)
                BirdClassMap[c] = i
                classPath = dataPath + c + '/'
                for path in Path(classPath).glob('*'):
                    chunks = dataProcessor.loadAudio(path)
                    for chunk in chunks:
                        folder = trainTestCalib(ratioTrainTestCalib)
                        tensorPath = destination + folder + str(ind) + '_' + stri + '.pt'
                        # torch.save(torch.from_numpy(dataProcessor.processChunk(chunk)),tensorPath)
                        indDict[folder].append(ind)
                        filesMap[ind] = tensorPath
                        classMap[ind] = i
                        ind += 1
            return BirdClassMap , indDict['train/'] , indDict['test/'] , indDict['calib/'] , filesMap , classMap


### TEST ###

# audio_file = r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\Birdclef2024\train_audio\asiope1\XC397761.ogg' #hop length aug -> x dimniue
# chunks, samplerate = audio_to_chunks(audio_file)
# print(len(chunks))
# S = chunk_to_spectrum(chunks[0], samplerate=samplerate, hop_length=716)
# print(S.shape)
