from dataLoading import CustomDataLoader
from myModels import AutoEncoder
from ULite import ULite
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
import time
from torchsummary import summary
from abc import ABC, abstractmethod

class Trainer(ABC):

    def __init__(self,model,dataLoader,nClasses,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath):

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

        self.nClasses = nClasses

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

class PreTrainer(Trainer):

    def __init__(self,model,dataLoader,nClasses,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath):

        super().__init__(model,dataLoader,nClasses,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath)

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

        # summary(self.model, (1, 224, 224))
        # print(self.model)

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

    def __init__(self,model,dataLoader,nClasses,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath):

        super().__init__(model,dataLoader,nClasses,nEpochs,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingPath)

    def test(self):

        # overallAccuracy = MulticlassAccuracy(num_classes=self.nClasses)
        # classAccuracy = MulticlassAccuracy(num_classes=self.nClasses, average=None)
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
                loss =+ self.criterion(Yhat, Y)
                # print('prediction')
                # print(Yhat)
                # print('groundtruth')
                # print(Y)
                # print('batch overall accuracy : ', self.overallAccuracy(Yhat,Y))
                # print('batch per label accuracy : ', self.classAccuracy(Yhat,Y))
                # acc += self.overallAccuracy(Yhat,Y)
                # classAcc += self.classAccuracy(Yhat,Y)
        end = time.time()
        print('testing time : ',(end-start)/3600)
        print('loss : ',loss/nBatch)
        print('overall accuracy : ', acc/nBatch)
        print('per label accuracy : ', classAcc/nBatch)

    def train(self):

        start = time.time()

        if (self.use_gpu):
            self.model.to('cuda:0')

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
                # print(batchHat.shape)
                # print(batchHat)
                # print(Y.shape)
                # print(Y)
                loss = self.criterion(Yhat, Y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            end = time.time()
            print('training time : ',(end-start)/3600)
            torch.save(self.model.state_dict(), self.savingPath + 'epoch_' + str(epoch))
            self.test()