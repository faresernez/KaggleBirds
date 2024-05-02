from pathlib import Path
import random
import torch
from utils import audio_to_chunks, chunk_to_spectrum, trainTestCalib, extract
from abc import ABC, abstractmethod
from dataProcessing import melSpectrogram
    

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

   
class ClassificationDataLoader(CustomDataLoader): #extracts samples from all audios randomly

    def __init__(self,batchSize,dataProcessor,ratioTrainTestCalib,dataPath,classes,dtype,extractionDone):
        
        super().__init__(batchSize,dataProcessor,ratioTrainTestCalib,dataPath,classes,dtype)

        self.extractionDone = extractionDone

        self.classFilesMap = {}

        self.BirdClassMap, self.trainIndex , self.testIndex , self.calibIndex , self.filesMap , self.classMap = extract(self.dataProcessor,self.ratioTrainTestCalib,self.dataPath,'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/finetuning/',self.classes,self.extractionDone)

        self.nFiles = len(self.filesMap)
        self.index = [*range(self.nFiles)]

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
    
    def reinitDataLoader(self,type):
        self.posResumeInIndex = 0

    
class OldClassificationDataLoader(CustomDataLoader): #extracts a batch from one audio at a time

    def __init__(self,batchSize,dataProcessor,ratioTrainTestCalib,dataPath,classes,dtype):
        
        super().__init__(batchSize,dataProcessor,ratioTrainTestCalib,dataPath,classes,dtype)

        self.classFilesMap = {}
        self.BirdClassMap = {}

        j = 0                
        for i,c in enumerate(classes):
            self.BirdClassMap[c] = i
            classPath = dataPath + c + '/'
            for path in Path(classPath).glob('*'):
                self.filesMap[j] = path
                self.classFilesMap[j] = i
                j+=1

        self.nFiles = len(self.filesMap)
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
        self.nCalibFiles = len(self.calibIndex)

    def loadBatch(self):
        nDataPoints = 0
        batchUncomplete = True
        batchX = torch.empty((0,1) + self.tensorShape, dtype = self.dtype)
        batchY = torch.empty((0), dtype = torch.long)

        while (batchUncomplete and self.posResumeInIndex < self.nFiles):
            chunks = self.dataProcessor.loadAudio(self.filesMap[self.index[self.posResumeInIndex]])
            nChunks = len(chunks)
            if ((nChunks - self.posResumeInFile) <= (self.batchSize - nDataPoints)):
                for chunk in chunks[self.posResumeInFile:]:
                    batchX = torch.cat((batchX,torch.from_numpy(self.dataProcessor.processChunk(chunk)).unsqueeze(0).unsqueeze(0)),dim = 0)
                y = torch.ones(len(chunks[self.posResumeInFile:]),dtype=torch.long)*self.classFilesMap[self.index[self.posResumeInIndex]]
                batchY = torch.cat((batchY,y),dim = 0)
                self.posResumeInIndex += 1
                self.posResumeInFile = 0
                nDataPoints = len(batchX)
            else:
                newPos = self.posResumeInFile + (self.batchSize - nDataPoints)
                for chunk in chunks[self.posResumeInFile:newPos]:
                    batchX = torch.cat((batchX,torch.from_numpy(self.dataProcessor.processChunk(chunk)).unsqueeze(0).unsqueeze(0)),dim = 0)
                y = torch.ones(len(chunks[self.posResumeInFile:newPos]),dtype=torch.long)*self.classFilesMap[self.index[self.posResumeInIndex]]
                batchY = torch.cat((batchY,y),dim = 0)
                self.posResumeInFile = newPos
                nDataPoints = len(batchX)
            if (nDataPoints == self.batchSize):
                batchUncomplete = False

        return batchX, batchY, self.posResumeInIndex < self.nFiles
    
### TESTS ###

# dataProcessor = melSpectrogram(seconds=5,
#                                sr=32000,
#                                n_mels=224,
#                                hop_length=716)

# DataLoader = OldClassificationDataLoader(batchSize=24,
#                                       dataProcessor = dataProcessor,
#                                       ratioTrainTestCalib=[0.8,0.2,0.],
#                                       dataPath= 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/train_audio/',
#                                       classes = ['ashpri1','barfly1','gloibi'],
#                                       dtype=torch.float32)

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