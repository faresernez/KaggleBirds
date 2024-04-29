from dataLoading import ClassificationDataLoader
from myModels import AutoEncoder, Classifier, ClassifierForULite
from ULite import ULite
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
import time
from torchmetrics.classification import MulticlassAccuracy

class Finetuner():

    def __init__(self,pretrainedModel,model,nClasses,inputSize,nEpochs,dataLoader,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingFile):

        self.pretrainedModel = pretrainedModel
        self.model = model
        self.nClasses = nClasses
        self.inputSize = inputSize
        self.nEpochs = nEpochs
        self.dataLoader = dataLoader
        self.criterion = criterion
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = use_gpu
        self.savingPath = savingPath
        self.loggingFile = loggingFile

        self.overallAccuracy = MulticlassAccuracy(num_classes=self.nClasses).to('cuda:0')
        self.classAccuracy = MulticlassAccuracy(num_classes=self.nClasses, average=None).to('cuda:0')

    def test(self):

        # overallAccuracy = MulticlassAccuracy(num_classes=self.nClasses)
        # classAccuracy = MulticlassAccuracy(num_classes=self.nClasses, average=None)


        start = time.time()
        nBatch = 0
        loss = 0
        self.dataLoader.startTesting()
        testing = True
        while testing:
            batchX, batchY, testing = self.dataLoader.loadBatch()
            nBatch += 1
            if self.use_gpu:
                batchX = batchX.to('cuda:0')
                batchY = batchY.to('cuda:0')
            with torch.no_grad():
                batchHat = self.model(batchX)
                loss =+ self.criterion(batchHat, batchY)
        end = time.time()
        print('testing time : ',(end-start)/3600)
        print('loss : ',loss/nBatch)
        print('overall accuracy : ', self.overallAccuracy(batchHat,batchY))
        print('per label accuracy : ', self.classAccuracy(batchHat,batchY))

    def train(self):

        start = time.time()

        if (self.use_gpu):
            self.model.to('cuda:0')

        for epoch in range(self.nEpochs):
            print('epoch : ', epoch)
            self.dataLoader.startTraining()
            training = True
            while training:
                batchX, batchY, training = self.dataLoader.loadBatch()
                if self.use_gpu:
                    batchX = batchX.to('cuda:0')
                    batchY = batchY.to('cuda:0')
                batchHat = self.model(batchX)
                self.optimizer.zero_grad()
                # print(batchHat.shape)
                # print(batchHat)
                # print(batchY.shape)
                # print(batchY)
                loss = self.criterion(batchHat, batchY)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            end = time.time()
            print('training time : ',(end-start)/3600)
            torch.save(self.model.state_dict(), self.savingPath + 'epoch_' + str(epoch))
            self.test()

# classes = ['ashpri1','barfly1','gloibi']
classes = ['houspa','grewar3','commyn']
tensorShape = (128,)

dataLoader  = ClassificationDataLoader(batchSize=48,
                trainingPercentage=0.8,
                dataPath= 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/train_audio/' ,
                classes=classes,
                tensorShape=(224,224),
                dtype=torch.float32,
                steps_per_subtrack=160000,
                samplerate=32000,
                n_mels=224,
                hop_length=716,
                fmax=16000)

pretrainedModelPath = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/BirdClef2024/pretrainedModel2/epoch_2' #ancien 3
pretrainedModel = ULite() #ULite
# pretrainedModel = AutoEncoder() #ULite
pretrainedModel.load_state_dict(torch.load(pretrainedModelPath))

nClasses = len(classes)
inputSize = 128

model = ClassifierForULite(pretrainedModel,inputSize,nClasses)

nEpochs = 100
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                step_size = 1, # Period of learning rate decay #ancien 5
                gamma = 0.7)
use_gpu = True
savingPath = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/BirdClef2024/finetunedModel2/'
loggingFile = None

finetuner = Finetuner(pretrainedModel=pretrainedModel,model=model,nClasses=nClasses,inputSize=inputSize,
                      nEpochs=nEpochs,dataLoader=dataLoader,criterion=criterion,lr=lr,optimizer=optimizer,scheduler=scheduler,use_gpu=use_gpu,savingPath=savingPath,loggingFile=loggingFile)
finetuner.train()
