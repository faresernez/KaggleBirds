import yaml
from training import Experience

def main():

    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Failed to read or parse the config file: {e}")
        return config
    
    config = config['config']

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
                    expOpts=expOpts)
    exp.launchExp()
    exp.reinitialize()

if __name__ == '__main__': 
    main()