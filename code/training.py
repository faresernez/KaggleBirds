# contains PreTrainer(), Finetuner(), FinetunerOptuna() and FinetunerForPyDataLoader()
# All classes are derived from Trainer() which initializes the trainer parameters and
# implements the train and test methods 

from dataProcessing import melSpectrogram
from dataLoading import PreTrainingDataLoader,  PytorchClassificationDataLoader, ClassificationDataLoader
from myModels import ClassifierForULite
from ULite import ULite
from utils import computeTrainingWeights, computeTrainingWeightsForPyDataLoader
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
import time
from torchsummary import summary
from abc import ABC, abstractmethod
from torchmetrics.classification import MulticlassAccuracy
import optuna



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

        if (self.use_gpu):
            self.model.to('cuda:0')

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

        return None

class Finetuner(Trainer): #only implemented for customDataLoader

    def __init__(self,model,dataLoader,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath):

        super().__init__(model,dataLoader,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath)

        self.overallAccuracy = MulticlassAccuracy(num_classes=self.nClasses).to('cuda:0')
        self.classAccuracy = MulticlassAccuracy(num_classes=self.nClasses, average=None).to('cuda:0')

    def test(self):

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
        
        accuracy = acc/nBatch
        classAcc = classAcc/nBatch
        loss = loss/nBatch
        accuracy = accuracy.cpu().numpy().tolist()
        classAcc = classAcc.cpu().numpy().tolist()
        loss = loss.cpu().numpy().tolist()
        
        return loss, accuracy, classAcc
    
    def train(self):

        if (self.use_gpu):
            self.model.to('cuda:0')

        # summary(self.model, (1, 224, 224))
        # print(self.model)

        for epoch in range(self.nEpochs):
            # print('epoch : ', epoch)
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

            if self.dataLoader.ratioTrainTestCalib[1] != 0:
                loss, accuracy, classAcc = self.test()

        if self.savingPath is not None:
            torch.save(self.model.state_dict(), self.savingPath + 'epoch_' + str(epoch))

        if self.dataLoader.ratioTrainTestCalib[1] != 0:
                return loss, accuracy, classAcc
        else:
            return None, None, None
        
        #log accuracy
        
class FinetunerOptuna(Finetuner): #to test
# class FinetunerOptuna(Trainer):

    def __init__(self,model,dataLoader,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath):

        super().__init__(model,dataLoader,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath)

        # self.overallAccuracy = MulticlassAccuracy(num_classes=self.nClasses).to('cuda:0')
        # self.classAccuracy = MulticlassAccuracy(num_classes=self.nClasses, average=None).to('cuda:0')

    # def test(self):

    #     nBatch = 0
    #     loss = 0
    #     acc = torch.tensor(0.).to('cuda:0')
    #     classAcc = torch.zeros(self.nClasses).to('cuda:0')
    #     self.dataLoader.startTesting()
    #     testing = True
    #     while testing:
    #         X, Y, testing = self.dataLoader.loadBatch()
    #         nBatch += 1
    #         if self.use_gpu:
    #             X = X.to('cuda:0')
    #             Y = Y.to('cuda:0')
    #         with torch.no_grad():
    #             Yhat = self.model(X)
    #             loss = self.criterion(Yhat, Y)
    #             acc += self.overallAccuracy(Yhat,Y)
    #             classAcc += self.classAccuracy(Yhat,Y)

    #     accuracy = acc/nBatch
    #     classAcc = classAcc/nBatch
    #     loss = loss/nBatch
    #     accuracy = accuracy.cpu().numpy().tolist()
    #     classAcc = classAcc.cpu().numpy().tolist()
    #     loss = loss.cpu().numpy().tolist()
        
    #     return loss, accuracy, classAcc

    def test(self):
        return super().test()

    def train(self,trial):
        
        for epoch in range(self.nEpochs):
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

            loss, accuracy, classAcc = self.test()
            trial.report(accuracy,epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        with open(self.loggingPath, 'a') as f:
            f.write('  results of the model on trial number : ' + str(trial.number))
            f.write('\n')
            f.write("  loss: " + str(loss))
            f.write('\n')
            f.write("  overall accuracy: " + str(accuracy))
            f.write('\n')
            f.write("  per class accuracy : " + str(classAcc))
            f.write('\n')

        return loss, accuracy, classAcc
    
class FinetunerOptunaForPyDataLoader(Trainer):

    def __init__(self,model,dataLoader,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath):

        super().__init__(model,dataLoader,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath)

        self.overallAccuracy = MulticlassAccuracy(num_classes=self.nClasses).to('cuda:0')
        self.classAccuracy = MulticlassAccuracy(num_classes=self.nClasses, average=None).to('cuda:0')

    def test(self):

        nBatch = 0
        loss = 0
        acc = torch.tensor(0.).to('cuda:0')
        classAcc = torch.zeros(self.nClasses).to('cuda:0')

        for X, Y in self.dataLoader.test_generator:
            nBatch += 1
            if self.use_gpu:
                X = X.to('cuda:0')
                Y = Y.to('cuda:0')
            with torch.no_grad():
                Yhat = self.model(X)
                loss = self.criterion(Yhat, Y)
                acc += self.overallAccuracy(Yhat,Y)
                classAcc += self.classAccuracy(Yhat,Y)

        accuracy = acc/nBatch
        classAcc = classAcc/nBatch
        loss = loss/nBatch
        accuracy = accuracy.cpu().numpy().tolist()
        classAcc = classAcc.cpu().numpy().tolist()
        loss = loss.cpu().numpy().tolist()
        
        return loss, accuracy, classAcc

    def train(self,trial):
        s = time.time()
        for epoch in range(self.nEpochs):
            s = time.time()
            for X, Y in self.dataLoader.train_generator:
                print('for Bla bla', time.time() - s)
                X, Y = X.to('cuda:0'), Y.to('cuda:0')
                # print(X.shape)
                # e = time.time()
                # print('loading a batch : ',e-et)
                Yhat = self.model(X)
                self.optimizer.zero_grad()
                loss = self.criterion(Yhat, Y)
                loss.backward()
                self.optimizer.step()
                # et = time.time()
                # print('training a batch : ',et-e)
            self.scheduler.step()
            # print('epoch finished : ',time.time() - s)

            s = time.time()
            print('start test')
            loss, accuracy, classAcc = self.test()
            print('test : ',time.time() - s)
            # s = time.time()
            trial.report(loss,epoch)
            # print('trial report : ',time.time() - s)

            # s = time.time()
            if trial.should_prune():
                raise optuna.TrialPruned()
        #     print('trial should prune test : ',time.time() - s)
        # print('train : ',time.time() - s)
        #log accuracy    
        print('overall accuracy : ', accuracy.cpu().numpy().tolist())
        print('per label accuracy : ', classAcc.cpu().numpy().tolist())

        return loss , accuracy, classAcc

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

