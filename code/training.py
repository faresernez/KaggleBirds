from dataProcessing import melSpectrogram
from dataLoading import PreTrainingDataLoader, ClassificationDataLoader, OldClassificationDataLoader 
from myModels import AutoEncoder, Classifier, ClassifierForULite
from ULite import ULite

import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
import time
from torchsummary import summary
from abc import ABC, abstractmethod
from torchmetrics.classification import MulticlassAccuracy



class Trainer(ABC):

    def __init__(self,model,dataLoader,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath):

        self.model = model
        self.nEpochs = nEpochs
        self.dataLoader = dataLoader
        self.criterion = criterion
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = use_gpu
        self.savingPath = savingPath
        self.loggingPath = loggingPath

        self.nClasses = dataLoader.nClasses

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

class PreTrainer(Trainer):

    def __init__(self,model,dataLoader,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath):

        super().__init__(model,dataLoader,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath)

    def test(self):
        start = time.time()
        nBatch = 0
        self.dataLoader.startTesting()
        testing = True
        while testing:
            X, testing = self.dataLoader.loadBatch()
            nBatch += 1
            if self.use_gpu:
                X = X.to('cuda:0')
            with torch.no_grad():
                Xhat = self.model(X)
                loss =+ self.criterion(Xhat, X)
        end = time.time()
        print('testing time : ',(end-start)/3600)
        print('loss : ',loss/nBatch)

    def train(self):

        start = time.time()

        if (self.use_gpu):
            self.model.to('cuda:0')

        # summary(self.model, (1,224,224))
        
        for epoch in range(self.nEpochs):
            print('epoch : ', epoch)
            self.dataLoader.startTraining()
            training = True
            while training:
                X, training = self.dataLoader.loadBatch()
                if self.use_gpu:
                    X = X.to('cuda:0')
                Xhat = self.model(X)
                self.optimizer.zero_grad()
                loss = self.criterion(Xhat, X)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            end = time.time()
            print('training time : ',(end-start)/3600)
            torch.save(self.model.state_dict(), self.savingPath + 'epoch_' + str(epoch))
            self.test()

class Finetuner(Trainer):

    def __init__(self,model,dataLoader,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath):

        super().__init__(model,dataLoader,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath)

        self.overallAccuracy = MulticlassAccuracy(num_classes=self.nClasses).to('cuda:0')
        self.classAccuracy = MulticlassAccuracy(num_classes=self.nClasses, average=None).to('cuda:0')

    def test(self):

        start = time.time()

        nBatch = 0
        loss = 0
        acc = torch.tensor(0.).to('cuda:0')
        classAcc = torch.zeros(self.nClasses).to('cuda:0')
        self.dataLoader.startTesting()
        testing = True
        while testing:
            X, Y, testing = self.dataLoader.loadBatch()
            nBatch += 1
            if self.use_gpu:
                X = X.to('cuda:0')
                Y = Y.to('cuda:0')
            with torch.no_grad():
                Yhat = self.model(X)
                loss = self.criterion(Yhat, Y)
                acc += self.overallAccuracy(Yhat,Y)
                classAcc += self.classAccuracy(Yhat,Y)
        end = time.time()
        print('testing time : ',(end-start)/3600)
        print('loss : ',loss/nBatch)
        print('overall accuracy : ', acc/nBatch)
        print('per label accuracy : ', classAcc/nBatch)

    def train(self):

        start = time.time()

        if (self.use_gpu):
            self.model.to('cuda:0')

        # summary(self.model, (1, 224, 224))
        # print(self.model)

        for epoch in range(self.nEpochs):
            print('epoch : ', epoch)
            self.dataLoader.startTraining()
            training = True
            while training:
                X, Y, training = self.dataLoader.loadBatch()
                if self.use_gpu:
                    X = X.to('cuda:0')
                    Y = Y.to('cuda:0')
                Yhat = self.model(X)
                self.optimizer.zero_grad()
                loss = self.criterion(Yhat, Y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            end = time.time()
            print('training time : ',(end-start)/3600)
            torch.save(self.model.state_dict(), self.savingPath + 'epoch_' + str(epoch))
            self.test()

class Experience():

    def __init__(self,dataProcessorOpts, classesDict, dataLoaderOpts, pretrainedModelOpts, modelOpts, expOpts):

        self.dataProcessorOpts = dataProcessorOpts
        self.classesDict = classesDict
        self.dataLoaderOpts = dataLoaderOpts
        self.pretrainedModelOpts = pretrainedModelOpts
        self.modelOpts = modelOpts
        self.expOpts = expOpts

        if (self.dataProcessorOpts['type'] == 'melSpectrogram'):
            self.dataProcessor = melSpectrogram(seconds = self.dataProcessorOpts['seconds'],
                                                sr = self.dataProcessorOpts['sr'],
                                                n_mels = self.dataProcessorOpts['n_mels'],
                                                hop_length = self.dataProcessorOpts['hop_length'])
        else:
            print('wrong dataProcessor')
        
        if self.classesDict is not None:
            self.exp_type = 'finetuning'
            self.classes = []
            self.w = []
            for key, value in self.classesDict.items():
                self.classes.append(key)
                self.w.append(value)
            s = sum(self.w)
            self.w = [ele / s for ele in self.w]
            self.w = torch.FloatTensor(self.w).to('cuda:0')

        else:
            self.exp_type = 'pretraining'
            self.classes = None
        
        if (self.dataLoaderOpts['dtype'] == 'torch.float32'):
            self.dtype = torch.float32

        self.extractionDone = self.dataLoaderOpts['extractionDone']
        self.batchSize = self.dataLoaderOpts['batchSize']

        if (self.dataLoaderOpts['type'] == 'ClassificationDataLoader'):
            self.dataLoader = ClassificationDataLoader(batchSize = self.batchSize,
                                                       dataProcessor = self.dataProcessor,
                                                       ratioTrainTestCalib = self.dataLoaderOpts['ratioTrainTestCalib'],
                                                       dataPath = self.dataLoaderOpts['dataPath'],
                                                       classes = self.classes,
                                                       dtype = self.dtype,
                                                       extractionDone = self.extractionDone)
        elif (self.dataLoaderOpts['type'] == 'OldClassificationDataLoader'):
            self.dataLoader = OldClassificationDataLoader(batchSize = self.dataLoaderOpts['batchSize'],
                                                    dataProcessor = self.dataProcessor,
                                                    ratioTrainTestCalib = self.dataLoaderOpts['ratioTrainTestCalib'],
                                                    dataPath = self.dataLoaderOpts['dataPath'],
                                                    classes = self.classes,
                                                    dtype =self.dtype)
            
        elif (self.dataLoaderOpts['type'] == 'PreTrainingDataLoader'):
            self.dataLoader = PreTrainingDataLoader(batchSize = self.dataLoaderOpts['batchSize'],
                                                    dataProcessor = self.dataProcessor,
                                                    ratioTrainTestCalib = self.dataLoaderOpts['ratioTrainTestCalib'],
                                                    dataPath = self.dataLoaderOpts['dataPath'],
                                                    classes = None,
                                                    dtype = self.dtype)
        else:
            print('wrong dataLoader')

        if (self.exp_type == 'finetuning'):
            if (self.pretrainedModelOpts['type'] == 'ULite'):
                self.pretrainedModel = ULite()
                self.pretrainedModel.load_state_dict(torch.load(self.pretrainedModelOpts['pretrainedModelPath']))
            else:
                print('wrong pretrained model')

            if (self.modelOpts['type'] == 'ClassifierForULite'):
                self.model = ClassifierForULite(self.pretrainedModel,self.dataLoader.nClasses)
            else:
                print('wrong model')

            if (self.modelOpts['criterion'] == 'CrossEntropyLoss'):
                self.criterion = nn.CrossEntropyLoss(weight = self.w)
            else:
                print('wrong criterion')
                
        else:
            if (self.modelOpts['type'] == 'ULite'):
                self.model = ULite()
            else:
                print('wrong model')

            if (self.modelOpts['criterion'] == 'MSELoss'):
                self.criterion = nn.MSELoss()
            else:
                print('wrong criterion')

        self.nEpochs = self.modelOpts['nEpochs']

        self.lr = self.modelOpts['lr']
        self.step_size = self.modelOpts['step_size']
        self.gamma = self.modelOpts['gamma']
        self.initializeOptimizer()
        # self.initializeScheduler()
        
        self.use_gpu = self.modelOpts['use_gpu']
        self.savingPath = self.expOpts['savingPath']
        self.loggingPath = self.expOpts['loggingPath']

        self.initialize()

    def initialize(self):
        if (self.exp_type == 'finetuning'):
            self.trainer = Finetuner(model=self.model,
                                     dataLoader=self.dataLoader,
                                     nEpochs=self.nEpochs,
                                     criterion=self.criterion,
                                     lr=self.lr,
                                     optimizer=self.optimizer,
                                     scheduler=self.scheduler,
                                     use_gpu=self.use_gpu,
                                     savingPath=self.savingPath,
                                     loggingPath=self.loggingPath)
        else:
            self.trainer = PreTrainer(model=self.model,
                                      dataLoader=self.dataLoader,
                                      nEpochs=self.nEpochs,
                                      criterion=self.criterion,
                                      lr=self.lr,
                                      optimizer=self.optimizer,
                                      scheduler=self.scheduler,
                                      use_gpu=self.use_gpu,
                                      savingPath=self.savingPath,
                                      loggingPath=self.loggingPath)
            
    def initializeDataLoader(self,type='train'):
        if type == 'train':
            self.dataLoader.startTraining()
        elif type == 'test':
            self.dataLoader.startTesting()
        elif type == 'calib':
            self.dataLoader.startCalibrating()

    def initializeScheduler(self):
        self.scheduler =  torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size = self.step_size,
                                                          gamma = self.gamma)

    def initializeOptimizer(self):
        if (self.modelOpts['optimizer'] == 'Adam'):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            print('wrong optimizer')
        self.initializeScheduler()

    def reinitialize(self):
        self.initializeDataLoader()
        self.initialize()

    def launchExp(self):

        print('bla bla bla') #or log bla bla bla
        self.trainer.train()
        print('fin bla bla bla') #or log bla bla bla


            
### TEST ###

# dataProcessorOpts = {'type' : 'melSpectrogram',
#                      'seconds' : 5,
#                      'sr' : 32000,
#                      'n_mels' : 224,
#                      'hop_length' : 716}

# # classesDict = {'nilfly2': 1.830038070678711, 
# #                 'rutfly6': 1.8044624328613281,
# #                 'scamin3': 1.7079267501831055, 
# #                 'bncwoo3': 1.6283693313598633, 
# #                 'wbbfly1': 1.2158184051513672, 
# #                 'pomgrp2': 1.0582923889160156, 
# #                 'inpher1': 0.9841403961181641, 
# #                 'blaeag1': 0.9221534729003906, 
# #                 'darter2': 0.6037435531616211,
# #                 'integr': 0.5658245086669922,
# #                 'asiope1': 0.5091238021850586,
# #                 'niwpig1': 0.3805961608886719}

# classesDict = None

# dataLoaderOpts = {'type' : 'PreTrainingDataLoader', #ClassificationDataLoader
#                 'batchSize' : 24,
#                 'ratioTrainTestCalib' : [0.8,0.2,0.],
#                 'dataPath' : 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/unlabeled_soundscapes/',
#                 # 'dataPath' : 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/train_audio/',
#                 'dtype' : 'torch.float32'} #ne9sin zouz

# pretrainedModelOpts = {'type' : 'ULite',
#                        'pretrainedModelPath' : 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/BirdClef2024/pretrainedModel3/epoch_4'}

# # modelOpts = {'type' : 'ClassifierForULite',
# #              'lr' : 0.001,
# #              'step_size' : 5,
# #              'gamma' : 0.8,
# #              'nEpochs' : 100,
# #              'optimizer' : 'Adam',
# #              'criterion' : 'CrossEntropyLoss',
# #              'use_gpu' : True
# #              }

# modelOpts = {'type' : 'ULite',
#              'lr' : 0.001,
#              'step_size' : 5,
#              'gamma' : 0.8,
#              'nEpochs' : 100,
#              'optimizer' : 'Adam',
#              'criterion' : 'MSELoss',
#              'use_gpu' : True
#              }

# # expOpts = {'savingPath' : 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/BirdClef2024/finetunedModel3/',
# expOpts = {'savingPath' : 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/BirdClef2024/pretrainedmodel3/',
#            'loggingPath' : None}

# exp = Experience(dataProcessorOpts, classesDict, dataLoaderOpts, pretrainedModelOpts, modelOpts, expOpts)
# exp.launchExp()

