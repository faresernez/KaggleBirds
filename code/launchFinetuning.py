from dataProcessing import melSpectrogram
from dataLoading import PreTrainingDataLoader, ClassificationDataLoader, OldClassificationDataLoader 
from myModels import AutoEncoder, Classifier, ClassifierForULite
from ULite import ULite
from training import Finetuner

import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
import time
from torchsummary import summary

pretrainedModelPath = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/BirdClef2024/pretrainedModel2/epoch_2' #ancien 3
pretrainedModel = ULite() #ULite AutoEncoder()
pretrainedModel.load_state_dict(torch.load(pretrainedModelPath))

dataProcessor = melSpectrogram(seconds=5,
                                sr=32000,
                                n_mels=224,
                                hop_length=716)

classes = ['ashpri1','barfly1','gloibi'] # 20mb ['houspa','grewar3','commyn'] 200mb
nClasses = len(classes) 


dataLoader = ClassificationDataLoader(batchSize=48,
                                      dataProcessor = dataProcessor,
                                      ratioTrainTestCalib=[0.8,0.2,0.],
                                      dataPath= 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/train_audio/',
                                      classes = classes,
                                      tensorShape=(224,224),
                                      dtype=torch.float32)

nEpochs = 100
criterion = nn.CrossEntropyLoss()
lr = 0.001
use_gpu = True

savingPath = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/BirdClef2024/pretrainedModel3/'
loggingPath = None

model = ClassifierForULite(pretrainedModel,nClasses) #ULite()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, 
                                            step_size = 1,
                                            gamma = 0.9)

trainer = Finetuner(model=model,dataLoader=dataLoader,nClasses=nClasses,nEpochs=nEpochs,criterion=criterion,lr=lr,optimizer=optimizer,scheduler=scheduler,use_gpu=use_gpu,savingPath=savingPath,loggingPath=loggingPath)
trainer.train()



