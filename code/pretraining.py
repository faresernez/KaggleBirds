from dataLoading import CustomDataLoader
from myModels import AutoEncoder
from ULite import ULite
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
import time
from torchsummary import summary


class Trainer():

    def __init__(self,model,nEpochs,dataLoader,criterion,lr,optimizer,scheduler,use_gpu,savingPath,loggingFile):

        self.model = model
        self.nEpochs = nEpochs
        self.dataLoader = dataLoader
        self.criterion = criterion
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = use_gpu
        self.savingPath = savingPath
        self.loggingFile = loggingFile

    def test(self):
        start = time.time()
        nBatch = 0
        self.dataLoader.startTesting()
        testing = True
        while testing:
            batch, testing = self.dataLoader.loadBatch()
            nBatch += 1
            if self.use_gpu:
                batch = batch.to('cuda:0')
            with torch.no_grad():
                batchHat = self.model(batch)
                loss =+ self.criterion(batchHat, batch)
        end = time.time()
        print('testing time : ',(end-start)/3600)
        print('loss : ',loss/nBatch)
        
    def train(self):

        start = time.time()

        if (self.use_gpu):
            self.model.to('cuda:0')

        summary(self.model, (1, 224, 224))
        print(self.model)

        for epoch in range(self.nEpochs):
            print('epoch : ', epoch)
            self.dataLoader.startTraining()
            training = True
            while training:
                batch, training = self.dataLoader.loadBatch()
                if self.use_gpu:
                    batch = batch.to('cuda:0')
                batchHat = self.model(batch)
                self.optimizer.zero_grad()
                loss = self.criterion(batchHat, batch)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            end = time.time()
            print('training time : ',(end-start)/3600)
            torch.save(self.model.state_dict(), self.savingPath + 'epoch_' + str(epoch))
            self.test()
            
dataLoader = CustomDataLoader(batchSize=48, #ancien 192
                              trainingPercentage=0.8,
                              dataPath= r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\Birdclef2024\unlabeled_soundscapes',
                              tensorShape=(224,224),
                              dtype=torch.float32,
                              steps_per_subtrack=160000,
                              samplerate=32000,
                              n_mels=224,
                              hop_length=716,
                              fmax=16000)

# model = AutoEncoder()
# nEpochs = 100
# criterion = nn.MSELoss()
# lr = 0.001
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
#                 step_size = 1, # Period of learning rate decay
#                 gamma = 0.5)
# use_gpu = True
# savingPath = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/BirdClef2024/pretrainedModel0/'
# loggingFile = None

# trainer = Trainer(model=model,nEpochs=nEpochs,dataLoader=dataLoader,criterion=criterion,lr=lr,optimizer=optimizer,scheduler=scheduler,use_gpu=use_gpu,savingPath=savingPath,loggingFile=loggingFile)
# trainer.train()

model = ULite()
nEpochs = 100
criterion = nn.MSELoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                step_size = 1, # Period of learning rate decay
                gamma = 0.9)
use_gpu = True
savingPath = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/BirdClef2024/pretrainedModel2/'
loggingFile = None

trainer = Trainer(model=model,nEpochs=nEpochs,dataLoader=dataLoader,criterion=criterion,lr=lr,optimizer=optimizer,scheduler=scheduler,use_gpu=use_gpu,savingPath=savingPath,loggingFile=loggingFile)
trainer.train()








