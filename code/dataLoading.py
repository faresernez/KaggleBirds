from pathlib import Path
import random
import librosa
import torch
import numpy as np


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

def chunk_to_spectrum(chunk,samplerate,n_mels=224, hop_length=512, fmax=16000): # S.shape = (X,224,313)
    # S = []
    # for chunk in chunks:
    S = librosa.feature.melspectrogram(y=chunk, sr=samplerate, n_mels=n_mels, hop_length=hop_length, fmax=fmax)
    # mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S))
    # print('shape of mfccs : ', S.shape) #(224,313)
    return S

# audio_file = r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\Birdclef2024\train_audio\asiope1\XC397761.ogg' #hop length aug -> x dimniue
# chunks, samplerate = audio_to_chunks(audio_file)
# print(len(chunks))
# S = chunk_to_spectrum(chunks[0], samplerate=samplerate, hop_length=716)
# print(S.shape)


class CustomDataLoader():

    def __init__(self,batchSize,trainingPercentage,dataPath,tensorShape,dtype,steps_per_subtrack,samplerate,n_mels,hop_length,fmax):

        self.steps_per_subtrack = steps_per_subtrack
        self.samplerate = samplerate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.fmax = fmax

        self.batchSize = batchSize
        self.trainingPercentage = trainingPercentage
        self.dataPath = dataPath
        self.filesMap = {}

        for i,path in enumerate(Path(dataPath).glob('*')):
            self.filesMap[i] = path

        self.nFiles = i+1
        self.index = [*range(self.nFiles)]

        self.trainIndex = []
        self.testIndex = []
        random.seed(10)
        for ind in self.index:  
            if (random.random() < self.trainingPercentage):
                self.trainIndex.append(ind)
            else:
                self.testIndex.append(ind)
        self.nTrainFiles = len(self.trainIndex)
        self.nTestFiles = len(self.testIndex)

        self.posResumeInIndex = 0
        self.posResumeInFile = 0
        self.tensorShape = tensorShape # Replace with auto value
        self.dtype = dtype

    def startEpoch(self):
        random.shuffle(self.index)
        self.posResumeInIndex = 0
        self.posResumeInFile = 0

    def startTraining(self):
        self.nFiles = self.nTrainFiles
        self.index = self.trainIndex
        random.shuffle(self.index)
        self.posResumeInIndex = 0
        self.posResumeInFile = 0
    
    def startTesting(self):
        self.nFiles = self.nTestFiles
        self.index = self.testIndex
        random.shuffle(self.index)
        self.posResumeInIndex = 0
        self.posResumeInFile = 0

    def loadBatch(self):
        nDataPoints = 0
        batchUncomplete = True
        batch = torch.empty((0,1) + self.tensorShape, dtype = self.dtype)
        while (batchUncomplete and self.posResumeInIndex < self.nFiles):
            chunks, samplerate = audio_to_chunks(self.filesMap[self.index[self.posResumeInIndex]],self.steps_per_subtrack,self.samplerate)
            nChunks = len(chunks)
            if ((nChunks - self.posResumeInFile) <= (self.batchSize - nDataPoints)):
                for chunk in chunks[self.posResumeInFile:]:
                    batch = torch.cat((batch,torch.from_numpy(chunk_to_spectrum(chunk,self.samplerate,self.n_mels,self.hop_length,self.fmax)).unsqueeze(0).unsqueeze(0)),dim = 0)
                    # batch.append(chunk_to_spectrum(chunk,samplerate))
                self.posResumeInIndex += 1
                self.posResumeInFile = 0
                nDataPoints = len(batch)
            else:
                newPos = self.posResumeInFile + (self.batchSize - nDataPoints)
                for chunk in chunks[self.posResumeInFile:newPos]:
                    batch = torch.cat((batch,torch.from_numpy(chunk_to_spectrum(chunk,samplerate,self.n_mels,self.hop_length,self.fmax)).unsqueeze(0).unsqueeze(0)),dim = 0)
                    # batch.append(chunk_to_spectrum(chunk,samplerate))
                self.posResumeInFile = newPos
                nDataPoints = len(batch)
            if (nDataPoints == self.batchSize):
                batchUncomplete = False
        return batch, self.posResumeInIndex < self.nFiles
    
class ClassificationDataLoader():

    def __init__(self,batchSize,trainingPercentage,dataPath,classes,tensorShape,dtype,steps_per_subtrack,samplerate,n_mels,hop_length,fmax):
        
        # data extractor variables
        self.steps_per_subtrack = steps_per_subtrack
        self.samplerate = samplerate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.fmax = fmax

        # input tensor variables
        self.tensorShape = tensorShape # Replace with auto value
        self.dtype = dtype

        # training variables || remarque : pk batch_size extraction = batch_size training ?
        self.batchSize = batchSize
        self.trainingPercentage = trainingPercentage

        # other variables
        self.dataPath = dataPath
        self.classes = classes

        # construct variables
        self.filesMap = {}
        self.classFilesMap = {}
        self.BirdClassMap = {}

        j = 0                
        for i,c in enumerate(classes):
            self.BirdClassMap[c] = i
            classPath = dataPath + c + '/'
            # print(classPath)
            for path in Path(classPath).glob('*'):
                self.filesMap[j] = path
                self.classFilesMap[j] = i
                j+=1

        self.nFiles = len(self.filesMap)
        self.index = [*range(self.nFiles)]

        self.trainIndex = []
        self.testIndex = []
        random.seed(10)

        for ind in self.index:  
            if (random.random() < self.trainingPercentage):
                self.trainIndex.append(ind)
            else:
                self.testIndex.append(ind)

        self.nTrainFiles = len(self.trainIndex)
        self.nTestFiles = len(self.testIndex)

        self.posResumeInIndex = 0
        self.posResumeInFile = 0

    def startEpoch(self):
        random.shuffle(self.index)
        self.posResumeInIndex = 0
        self.posResumeInFile = 0

    def startTraining(self):
        self.nFiles = self.nTrainFiles
        self.index = self.trainIndex
        random.shuffle(self.index)
        self.posResumeInIndex = 0
        self.posResumeInFile = 0
    
    def startTesting(self):
        self.nFiles = self.nTestFiles
        self.index = self.testIndex
        random.shuffle(self.index)
        self.posResumeInIndex = 0
        self.posResumeInFile = 0

    def loadBatch(self):
        nDataPoints = 0
        batchUncomplete = True
        batchX = torch.empty((0,1) + self.tensorShape, dtype = self.dtype)
        batchY = torch.empty((0), dtype = torch.long)
        while (batchUncomplete and self.posResumeInIndex < self.nFiles):
            chunks, samplerate = audio_to_chunks(self.filesMap[self.index[self.posResumeInIndex]],self.steps_per_subtrack,self.samplerate)
            nChunks = len(chunks)
            if ((nChunks - self.posResumeInFile) <= (self.batchSize - nDataPoints)):
                for chunk in chunks[self.posResumeInFile:]:
                    batchX = torch.cat((batchX,torch.from_numpy(chunk_to_spectrum(chunk,self.samplerate,self.n_mels,self.hop_length,self.fmax)).unsqueeze(0).unsqueeze(0)),dim = 0)
                    # batch.append(chunk_to_spectrum(chunk,samplerate))
                # print('shape de batchx')
                # print(batchX.shape)
                t = torch.ones(len(chunks[self.posResumeInFile:]),dtype=torch.long)*self.classFilesMap[self.index[self.posResumeInIndex]]
                # print('shape de t')
                # print(t.shape)
                # print(t)
                # print('shape batch y')
                # print(batchY.shape)
                batchY = torch.cat((batchY,t),dim = 0)
                # print('shape batchY apres cat')
                # print(batchY.shape)
                self.posResumeInIndex += 1
                self.posResumeInFile = 0
                nDataPoints = len(batchX)
            else:
                newPos = self.posResumeInFile + (self.batchSize - nDataPoints)
                for chunk in chunks[self.posResumeInFile:newPos]:
                    batchX = torch.cat((batchX,torch.from_numpy(chunk_to_spectrum(chunk,samplerate,self.n_mels,self.hop_length,self.fmax)).unsqueeze(0).unsqueeze(0)),dim = 0)
                    # batch.append(chunk_to_spectrum(chunk,samplerate))
                t = torch.ones(len(chunks[self.posResumeInFile:newPos]),dtype=torch.long)*self.classFilesMap[self.index[self.posResumeInIndex]]
                # print(t)
                batchY = torch.cat((batchY,t),dim = 0)
                self.posResumeInFile = newPos
                nDataPoints = len(batchX)
            if (nDataPoints == self.batchSize):
                batchUncomplete = False
        return batchX, batchY, self.posResumeInIndex < self.nFiles

DataLoader = ClassificationDataLoader(batchSize=24,
                                      trainingPercentage=0.8,
                                      dataPath= 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/train_audio/',
                                      classes = ['ashpri1','barfly1','gloibi'],
                                      tensorShape=(224,224),
                                      dtype=torch.float32,steps_per_subtrack=160000,samplerate=32000,n_mels=224,hop_length=716,fmax=16000)


# print(DataLoader.posResumeInIndex)
# print(DataLoader.posResumeInFile)
# for j in range(1999):
#     print('batch number : ', j)
#     DataLoader.startEpoch()
#     x , y , test = DataLoader.loadBatch()
#     print(DataLoader.posResumeInIndex)
#     print(DataLoader.posResumeInFile)
#     print(x.shape)
#     print(y.shape)


# DataLoader = CustomDataLoader(batchSize=24,dataPath= r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\Birdclef2024\unlabeled_soundscapes',tensorShape=(224,224),dtype=torch.float32,steps_per_subtrack=160000,samplerate=32000,n_mels=224,hop_length=716,fmax=16000)
# print(DataLoader.posResumeInIndex)
# print(DataLoader.posResumeInFile)
# for j in range(10):
#     print('batch number : ', j)
#     b = DataLoader.load_batch()
#     print(DataLoader.posResumeInIndex)
#     print(DataLoader.posResumeInFile)
#     print(b.shape)

# DataLoader = CustomDataLoader(48,r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\Birdclef2024\unlabeled_soundscapes')
# print(DataLoader.posResumeInIndex)
# print(DataLoader.posResumeInFile)
# for j in range(5):
#     print('batch number : ', j)
#     b = DataLoader.load_batch()
#     print(DataLoader.posResumeInIndex)
#     print(DataLoader.posResumeInFile)