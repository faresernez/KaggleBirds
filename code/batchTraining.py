# code to launch a hyperparameters search and train the model with the optimal ones for each classifier sequentially
# Automating the training

from experience import launchOptunaSearch
from experience import Experience, ExperienceOptuna
import yaml
from utils import extractSubgroups

def main():

    loggingPath = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/results/results2024/'
    savingPath = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/BirdClef2024/optunaModels/'

    groups = extractSubgroups()

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

    optunaParams0 = config['optunaParams0']
    optunaParams1 = config['optunaParams1']
    optunaParams2 = config['optunaParams2']

    for group, subgroups in enumerate(groups):

        # if group == 1: 
        #     break

        for subgroup, classes in enumerate(subgroups):
            print(subgroup)

            # if ( subgroup != 0 and subgroup != 2 and subgroup != 9 and subgroup != 10 and subgroup != 14):
            #     continue

            # if ( subgroup < 10):
            #     continue
             
            currentSavingPath = savingPath + str(group) + '_' + str(subgroup)
            currentLoggingPath = loggingPath + str(group) + '_' + str(subgroup) + '.txt'
            expOpts['savingPath'] = currentSavingPath
            expOpts['loggingPath'] = currentLoggingPath


            if subgroup < 2:
                optunaParams = optunaParams0 
            elif subgroup > 9:
                optunaParams = optunaParams2
            else:
                optunaParams = optunaParams1

            with open(currentLoggingPath, 'a') as f:
                f.write('starting study with group : ' + str(group) + ' subgroup : ' + str(subgroup))
                f.write('\n')

            exp_type = 'optunafinetuning'
            dataLoaderOpts['ratioTrainTestCalib'] = [0.8,0.2,0.]
            dataLoaderOpts['extractionDone'] = False


            modelParams , study_trials , pruned_trials , complete_trials , best_trial_value, best_trial_number, studyDuration = launchOptunaSearch(dataProcessorOpts,
                                                                                                                            classes,
                                                                                                                            dataLoaderOpts,
                                                                                                                            pretrainedModelOpts,
                                                                                                                            modelOpts,
                                                                                                                            expOpts,
                                                                                                                            exp_type,
                                                                                                                            optunaParams)
            
            with open(currentLoggingPath, 'a') as f:
                f.write("  Number of finished trials: " + str(study_trials))
                f.write('\n')
                f.write("  Number of pruned trials: "+ str(pruned_trials))
                f.write('\n')
                f.write("  Number of complete trials: "+ str(complete_trials))
                f.write('\n')

                f.write("Best trial:")
                f.write('\n')

                f.write("  Number: "+ str(best_trial_number))
                f.write('\n')

                f.write("  Value: "+ str(best_trial_value))
                f.write('\n')

                f.write("  Params: ")
                f.write('\n')
                for key, value in modelParams.items():
                    f.write("    {}: {}".format(key, value))
                    f.write('\n')

                f.write('  study duration: ' + str(studyDuration/3600))
                f.write('\n')

                f.write('starting finetuning')
                f.write('\n')

            exp_type = 'finetuning'
            dataLoaderOpts['ratioTrainTestCalib'] = [1.0,0.,0.]

            finetuningExp = Experience(dataProcessorOpts=dataProcessorOpts, 
                            classes=classes,
                            dataLoaderOpts=dataLoaderOpts, 
                            pretrainedModelOpts=pretrainedModelOpts, 
                            modelOpts=modelOpts, 
                            expOpts=expOpts,
                            exp_type=exp_type)
            
            finetuningExp.lr = modelParams['lr']
            finetuningExp.nEpochs = modelParams['Epochs']
            finetuningExp.gamma = modelParams['gamma']
            finetuningExp.step_size = modelParams['step_size']

            finetuningExp.initializeOptimizer()
            finetuningExp.reinitialize()
            finetuningExp.launchExp()

if __name__ == '__main__': 
    main()



