# contains the different data loaders that can be used during training

from pathlib import Path
import random
import torch
from utils import trainTestCalib, extract, newExtract
from abc import ABC, abstractmethod
from dataProcessing import melSpectrogram
import time
import torch


class CustomDataLoader(ABC):

    def __init__(self,batchSize,dataProcessor,ratioTrainTestCalib,dataPath,classes,dtype): 

        random.seed(10)

        self.dataProcessor = dataProcessor

        self.tensorShape = dataProcessor.tensorShape # Replace with auto value
        self.dtype = dtype

        self.classes = classes
        if (classes is not None):
            self.nClasses = len(classes)
        else:
            self.nClasses = None


        self.batchSize = batchSize
        self.ratioTrainTestCalib = ratioTrainTestCalib
        self.dataPath = dataPath

        self.nFiles = 0
        self.filesMap = {}
        self.index = []
        self.trainIndex = []
        self.testIndex = []
        self.calibIndex = []

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

    def startCalibrating(self):
        self.nFiles = self.nCalibFiles
        self.index = self.calibIndex
        random.shuffle(self.index)
        self.posResumeInIndex = 0
        self.posResumeInFile = 0

    @abstractmethod
    def loadBatch(self):
        pass

class PreTrainingDataLoader(CustomDataLoader):

    def __init__(self,batchSize,dataProcessor,ratioTrainTestCalib,classes,dataPath,dtype):

        super().__init__(batchSize,dataProcessor,ratioTrainTestCalib,dataPath,classes,dtype)

        self.nFiles = 0
        self.filesMap = {}
        self.index = []
        self.trainIndex = []
        self.testIndex = []
        
        self.posResumeInIndex = 0
        self.posResumeInFile = 0

        for i,path in enumerate(Path(dataPath).glob('*')):
            self.filesMap[i] = path

        self.nFiles = i+1
        self.index = [*range(self.nFiles)]

        for ind in self.index:
            dest = trainTestCalib(ratioTrainTestCalib)
            if (dest == 'train/'):
                self.trainIndex.append(ind)
            elif (dest == 'test/'):
                self.testIndex.append(ind)
            else:
                self.calibIndex.append(ind)

        self.nTrainFiles = len(self.trainIndex)
        self.nTestFiles = len(self.testIndex)


    def loadBatch(self):
        nDataPoints = 0
        batchUncomplete = True
        batch = torch.empty((0,1) + self.tensorShape, dtype = self.dtype)
        while (batchUncomplete and self.posResumeInIndex < self.nFiles):
            chunks = self.dataProcessor.loadAudio(self.filesMap[self.index[self.posResumeInIndex]])
            nChunks = len(chunks)
            if ((nChunks - self.posResumeInFile) <= (self.batchSize - nDataPoints)):
                for chunk in chunks[self.posResumeInFile:]:
                    batch = torch.cat((batch,torch.from_numpy(self.dataProcessor.processChunk(chunk)).unsqueeze(0).unsqueeze(0)),dim = 0)
                self.posResumeInIndex += 1
                self.posResumeInFile = 0
                nDataPoints = len(batch)
            else:
                newPos = self.posResumeInFile + (self.batchSize - nDataPoints)
                for chunk in chunks[self.posResumeInFile:newPos]:
                    batch = torch.cat((batch,torch.from_numpy(self.dataProcessor.processChunk(chunk)).unsqueeze(0).unsqueeze(0)),dim = 0)
                self.posResumeInFile = newPos
                nDataPoints = len(batch)
            if (nDataPoints == self.batchSize):
                batchUncomplete = False

        return batch, self.posResumeInIndex < self.nFiles
    
class ClassificationDataLoader(CustomDataLoader):

    def __init__(self,batchSize,dataProcessor,ratioTrainTestCalib,dataPath,classes,dtype,tensorShape,extractionDone,byFileExtraction=True):
        
        super().__init__(batchSize,dataProcessor,ratioTrainTestCalib,dataPath,classes,dtype)

        self.extractionDone = extractionDone
        self.byFileExtraction = byFileExtraction
        self.tensorShape = tensorShape

        destination = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/Birdclef2024/finetuning/'

        self.BirdClassMap , self.trainIndex , self.testIndex , self.calibIndex , self.filesMap , self.classMap = extract(self.dataProcessor,
                                                                                                                         self.ratioTrainTestCalib,
                                                                                                                         self.dataPath,
                                                                                                                         destination,
                                                                                                                         self.classes,
                                                                                                                         self.extractionDone)

        self.nTrainFiles = len(self.trainIndex)
        self.nTestFiles = len(self.testIndex)
        self.nCalibFiles = len(self.calibIndex)

    def loadBatch(self):

        nDataPoints = 0
        batchX = torch.empty((0,1) + self.tensorShape, dtype = self.dtype)
        batchY = torch.empty((0), dtype = torch.long)

        while (nDataPoints < self.batchSize and self.posResumeInIndex < self.nFiles):
            ind = self.index[self.posResumeInIndex]
            path = self.filesMap[ind]
            x = torch.load(path).unsqueeze(0).unsqueeze(0)
            batchX = torch.cat((batchX,x),dim = 0) #why not directly to gpu ?
            y = torch.ones(1,dtype=torch.long)*self.classMap[ind]
            batchY = torch.cat((batchY,y),dim = 0)
            self.posResumeInIndex += 1
            nDataPoints += 1

        return batchX, batchY, self.posResumeInIndex < self.nFiles
    
class PytorchClassificationDataLoader():  #A data loader based on torch.utils.data.DataLoader and torch.utils.data.Dataset

    def __init__(self,batchSize,dataProcessor,ratioTrainTestCalib,dataPath,classes,dtype,extractionDone):

        random.seed(10)

        self.batchSize = batchSize
        self.dataProcessor = dataProcessor
        self.ratioTrainTestCalib = ratioTrainTestCalib
        self.dataPath = dataPath
        self.classes = classes
        self.dtype = dtype
        self.extractionDone = extractionDone

        self.nClasses = len(classes)

        partition, labels = newExtract(dataProcessor,ratioTrainTestCalib,dataPath,classes,extractionDone)

        self.params = {'batch_size': self.batchSize, 'shuffle': True, 'num_workers': 4}

        if self.ratioTrainTestCalib[0] != 0 :
            self.train_set = Dataset(partition['train'], labels)
            self.train_generator = torch.utils.data.DataLoader(self.train_set, **self.params)
        if self.ratioTrainTestCalib[1] != 0 :
            self.test_set = Dataset(partition['test'], labels)
            self.test_generator = torch.utils.data.DataLoader(self.test_set, **self.params)
        if self.ratioTrainTestCalib[2] != 0 :
            self.calib_set = Dataset(partition['calib'], labels)
            self.calib_generator = torch.utils.data.DataLoader(self.calib_set, **self.params)
    
class Dataset(torch.utils.data.Dataset):

    def __init__(self, list_IDs, labels):

        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = torch.load('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/Birdclef2024/finetuning/' + ID + '.pt')
        y = self.labels[ID]

        return X, y

    
### TESTS ###

# if __name__ == '__main__':
#     dataProcessor = melSpectrogram(seconds=5,
#                                 sr=32000,
#                                 n_mels=224,
#                                 hop_length=716)

#     DataLoader = NewClassificationDataLoader(batchSize=48,
#                                         dataProcessor = dataProcessor,
#                                         ratioTrainTestCalib=[0.8,0.2,0.],
#                                         dataPath= 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/train_audio/',
#                                         classes = ['ashpri1','barfly1','gloibi'],
#                                         dtype=torch.float32,
#                                         extractionDone = False)

#     DataLoader.startTraining()
#     x,y,test = DataLoader.loadBatch()
#     print(x)
#     print(y)
#     print(x.shape)
#     print(y.shape)
#     x,y,test = DataLoader.loadBatch()
#     print(x)
#     print(y)
#     print(x.shape)
#     print(y.shape)

# for j in range(10):
#     print('batch number : ', j)
#     x , y , test = DataLoader.loadBatch()
#     # print(DataLoader.posResumeInIndex)
#     # print(DataLoader.posResumeInFile)
#     print(x.shape)
#     print(y.shape)


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


# DataLoader = PreTrainingDataLoader(batchSize=24,
#                                    dataProcessor = dataProcessor,
#                                    ratioTrainTestCalib=[0.8,0.2,0.],
#                                    dataPath= 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/unlabeled_soundscapes/',
#                                    classes = [],
#                                    dtype=torch.float32)

# print(DataLoader.posResumeInIndex)
# print(DataLoader.posResumeInFile)
# for j in range(10):
#     print('batch number : ', j)
#     b , test = DataLoader.loadBatch()
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