# code to launch the optimal hyperparameters search from a config file with optuna

from experience import  launchOptunaSearch
from torchmetrics.classification import MulticlassAccuracy
import torch
import torch.nn as nn
import time
torch.backends.cudnn.benchmark = True
import yaml

def main():

    params = {'reduction_factor' : 2,
            'min_early_stopping_rate' : 1,
            'n_trials' : 30,
            'timeout': 14400,
            }

    start = time.time()
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Failed to read or parse the config file: {e}")
        return config
    
    config = config['config']
    dataProcessorOpts = config['dataProcessorOpts']
    classes = config['classes']
    dataLoaderOpts = config['dataLoaderOpts']
    pretrainedModelOpts = config['pretrainedModelOpts']
    modelOpts = config['modelOpts']
    expOpts = config['expOpts']
    exp_type = config['exp_type']

    return launchOptunaSearch(dataProcessorOpts,classes,dataLoaderOpts,pretrainedModelOpts,modelOpts,expOpts,exp_type,params)