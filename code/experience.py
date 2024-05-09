# contains Experience(), ExperienceOptuna() and launchOptunaExp()
# ExperienceOptuna() inherits nearly everything from Experience(), 
# it adds a slight change in launchExp() to take into account the trial from the optuna study and implements the objective method*
# Experience class is the main class in this repo. It contains the attributes and implements the methods needed to run one or multiple trainings efficiently



from dataProcessing import melSpectrogram
from dataLoading import PreTrainingDataLoader,  PytorchClassificationDataLoader, ClassificationDataLoader
from myModels import ClassifierForULite
from ULite import ULite
from training import Finetuner, PreTrainer, FinetunerOptuna, FinetunerOptunaForPyDataLoader
from utils import computeTrainingWeights, computeTrainingWeightsForPyDataLoader
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
import time
from torchsummary import summary

import optuna
from optuna.trial import TrialState
from experience import  ExperienceOptuna

class Experience():

    def __init__(self,dataProcessorOpts, classes, dataLoaderOpts, pretrainedModelOpts, modelOpts, expOpts, exp_type):

        self.dataProcessorOpts = dataProcessorOpts
        self.classes = classes
        self.dataLoaderOpts = dataLoaderOpts
        self.pretrainedModelOpts = pretrainedModelOpts
        self.modelOpts = modelOpts
        self.expOpts = expOpts
        self.exp_type = exp_type

        if (self.dataProcessorOpts['type'] == 'melSpectrogram'):
            self.dataProcessor = melSpectrogram(seconds = self.dataProcessorOpts['seconds'],
                                                sr = self.dataProcessorOpts['sr'],
                                                n_mels = self.dataProcessorOpts['n_mels'],
                                                hop_length = self.dataProcessorOpts['hop_length'])
        else:
            print('wrong dataProcessor')

        if (self.dataLoaderOpts['dtype'] == 'torch.float32'):
            self.dtype = torch.float32

        self.extractionDone = self.dataLoaderOpts['extractionDone']
        self.byFileExtraction = self.dataLoaderOpts['byFileExtraction']
        self.batchSize = self.dataLoaderOpts['batchSize']

        if (self.dataLoaderOpts['type'] == 'PytorchClassificationDataLoader'):
            self.dataLoader = PytorchClassificationDataLoader(batchSize = self.batchSize,
                                                       dataProcessor = self.dataProcessor,
                                                       ratioTrainTestCalib = self.dataLoaderOpts['ratioTrainTestCalib'],
                                                       dataPath = self.dataLoaderOpts['dataPath'],
                                                       classes = self.classes,
                                                       dtype = self.dtype,
                                                       extractionDone = self.extractionDone)
            
        elif (self.dataLoaderOpts['type'] == 'ClassificationDataLoader'):
            self.dataLoader = ClassificationDataLoader(batchSize = self.batchSize,
                                                       dataProcessor = self.dataProcessor,
                                                       ratioTrainTestCalib = self.dataLoaderOpts['ratioTrainTestCalib'],
                                                       dataPath = self.dataLoaderOpts['dataPath'],
                                                       classes = self.classes,
                                                       dtype = self.dtype,
                                                       tensorShape=self.dataProcessor.tensorShape,
                                                       extractionDone = self.extractionDone)
            
        elif (self.dataLoaderOpts['type'] == 'PreTrainingDataLoader'):
            self.dataLoader = PreTrainingDataLoader(batchSize = self.dataLoaderOpts['batchSize'],
                                                    dataProcessor = self.dataProcessor,
                                                    ratioTrainTestCalib = self.dataLoaderOpts['ratioTrainTestCalib'],
                                                    dataPath = self.dataLoaderOpts['dataPath'],
                                                    classes = None,
                                                    dtype = self.dtype)
        else:
            print('wrong dataLoader')

        if (self.exp_type != 'pretraining'): #ici
            self.w = computeTrainingWeights(len(self.classes)).to('cuda:0') #computeTrainingWeightsForPyDataLoader
            print(self.w)

        if (self.exp_type != 'pretraining'):
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
        elif (self.exp_type == 'pretraining'):
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
        elif (self.exp_type == 'optunafinetuning'):
            self.trainer = FinetunerOptuna(model=self.model,
                                      dataLoader=self.dataLoader,
                                      nEpochs=self.nEpochs,
                                      criterion=self.criterion,
                                      lr=self.lr,
                                      optimizer=self.optimizer,
                                      scheduler=self.scheduler,
                                      use_gpu=self.use_gpu,
                                      savingPath=self.savingPath,
                                      loggingPath=self.loggingPath,)
                                    #   trial = self.trial)
        elif (self.exp_type == 'pydloptunafinetuning'):
            self.trainer = FinetunerOptunaForPyDataLoader(model=self.model,
                                      dataLoader=self.dataLoader,
                                      nEpochs=self.nEpochs,
                                      criterion=self.criterion,
                                      lr=self.lr,
                                      optimizer=self.optimizer,
                                      scheduler=self.scheduler,
                                      use_gpu=self.use_gpu,
                                      savingPath=self.savingPath,
                                      loggingPath=self.loggingPath,)
                                    #   trial = self.trial)
            
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
        self.initializeDataLoader() #ici
        self.initialize()

    def launchExp(self):
        return self.trainer.train()
    
class ExperienceOptuna(Experience):
    
    def __init__(self,dataProcessorOpts, classes, dataLoaderOpts, pretrainedModelOpts, modelOpts, expOpts, exp_type):

        super().__init__(dataProcessorOpts, classes, dataLoaderOpts, pretrainedModelOpts, modelOpts, expOpts, exp_type)

    def launchExp(self,trial):
        return self.trainer.train(trial)

    def objective(self,trial):

        newLR = trial.suggest_float("lr", 1e-6, 5e-4, log=True)
        newNEpochs = trial.suggest_int("Epochs", 5, 30, log=True)
        newGamma = trial.suggest_float("gamma", 0.9, 0.99, log=True)
        newStepSize = trial.suggest_int("step_size", 1, 2, log=True)

        self.lr = newLR
        self.nEpochs = newNEpochs
        self.gamma = newGamma
        self.step_size = newStepSize

        # self.dataLoader.batchSize = newBatchSize
        self.initializeOptimizer()
        self.reinitialize()
        loss , accuracy, classAcc = self.launchExpTrial(trial)
        return accuracy
    
def launchOptunaSearch(dataProcessorOpts,classes,dataLoaderOpts,pretrainedModelOpts,modelOpts,expOpts,exp_type,params):
    
    start = time.time()
    Optunaexp = ExperienceOptuna(dataProcessorOpts=dataProcessorOpts, 
                    classes=classes,
                    dataLoaderOpts=dataLoaderOpts, 
                    pretrainedModelOpts=pretrainedModelOpts, 
                    modelOpts=modelOpts, 
                    expOpts=expOpts,
                    exp_type=exp_type)
    
    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.HyperbandPruner(min_resource=1,
                                                                      max_resource=Optunaexp.nEpochs,
                                                                      reduction_factor=params['reduction_factor'],),)
                                                                    #   min_early_stopping_rate=params['min_early_stopping_rate']),)
    
    study.optimize(Optunaexp.objective, n_trials=params['n_trials'], timeout=params['timeout'])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    studyDuration = time.time() - start
    print('  study duration: ',studyDuration/3600)

    return trial.params , len(study.trials) , len(pruned_trials) , len(complete_trials) , trial.value, trial.number ,studyDuration