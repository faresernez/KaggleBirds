# code to launch multiple experiences from a config file

import yaml
from experience import Experience

def main():

    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Failed to read or parse the config file: {e}")
        return config
    
    config = config['config']

    exp_type = config[exp_type]

    dataProcessorOpts = config['dataProcessorOpts']

    classesDict = config['classesDict']

    dataLoaderOpts = config['dataLoaderOpts']

    pretrainedModelOpts = config['pretrainedModelOpts']

    modelOpts = config['modelOpts']

    expOpts = config['expOpts']

    exp = Experience(dataProcessorOpts=dataProcessorOpts, 
                    classesDict=classesDict,
                    dataLoaderOpts=dataLoaderOpts, 
                    pretrainedModelOpts=pretrainedModelOpts, 
                    modelOpts=modelOpts, 
                    expOpts=expOpts,
                    exp_type=exp_type)
    
    exp.launchExp()
    print('lr')
    print(exp.lr)
    exp.lr = 0.0001
    exp.initializeOptimizer()
    exp.reinitialize()
    exp.launchExp()
    print('new lr')
    print(exp.lr)

if __name__ == '__main__': 
    main()