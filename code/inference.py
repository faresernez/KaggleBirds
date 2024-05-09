# Test the inference before submitting a kaggle notebook

import glob
import os
import numpy as np
from dataProcessing import melSpectrogram
import torch
from ULite import ULite
from myModels import ClassifierForULite
import time
import pandas as pd
from utils import extractSubgroups , indices_of_top_values

# construct dictionary
# models classes
# fn to import and load the models
# maybe quantization
# 

softmax = torch.nn.Softmax(dim = 1)

def loadModels():
    models = []
    model_path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/models/BirdClef2024/optunaModels/'
    pretrainedModel = ULite()
    for group in range(2):
        for subgroup in range(12):
            if (group == 0 and subgroup == 0):
                model = ClassifierForULite(pretrainedModel,nClasses=17)
            elif (group == 1 and subgroup <= 1):
                model = ClassifierForULite(pretrainedModel,nClasses=16)
            else:
                model = ClassifierForULite(pretrainedModel,nClasses=15)

            path = model_path + str(group) + '_' + str(subgroup)           
            model.load_state_dict(torch.load(path))
            models.append(model)

    return models

def predict(model, nClasses, n_loops, batch_size, path):

    with torch.no_grad():
        ypred_test = torch.empty((0,nClasses), dtype = torch.long)
        for j in range(n_loops):
            t = torch.load(path + str(j*batch_size) + '.pt' , map_location='cpu').unsqueeze(0).unsqueeze(0)
            for i in range(batch_size - 1):
                    x = torch.load(path + str(j*batch_size + i + 1) + '.pt' , map_location='cpu').unsqueeze(0).unsqueeze(0)
                    t = torch.cat((t,x),dim=0)
            y_hat = model(t)
            ypred_test = torch.cat((ypred_test,y_hat),dim=0)
        
        ypred_test = softmax(ypred_test)
        ypred_test.cpu().numpy()
    
    return ypred_test


def predict_for_sample(filename, sample_sub, dataProcessor, models, groups, competition_classes):
    print(filename)
    print(filename.split(".ogg")[0].split("/"))
    file_id = filename.split(".ogg")[0].split("/")[-1] + '_'
    probabilities = np.ones((182,),dtype=np.float32)*4.798188e-08
    # os.makedirs('/kaggle/working/chunks/')
    # path = '/kaggle/working/chunks/'
    path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/submission_example/'
    chunks = dataProcessor.loadAudio(filename)
    for i,chunk in enumerate(chunks):
        # torch.save(dataProcessor.processChunk(chunk),'/kaggle/working/chunks/' + str(i) + '.pt')
        torch.save(torch.from_numpy(dataProcessor.processChunk(chunk)),'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/submission_example/' + str(i) + '.pt')
        
    del chunks

    q_hat = 0.9647314725735

    nChunks = len(os.listdir(path))
    batch_size = 8
    n_loops = nChunks // batch_size

    s = time.time()

    predictions = []
    predictionsGroup0 = []
    for _ in range(nChunks-1):
        predictions.append([])
        predictionsGroup0.append([])

    for group in range(2):
         for subgroup in range(12):

            training_list = groups[group][subgroup]

            if (group == 0 and subgroup == 0):
                nClasses=17
            elif (group == 1 and subgroup <= 1):
                nClasses=16
            else:
                nClasses=15
              
            modelInd = group*12 + subgroup
            models[modelInd].eval()

            ypred_test = predict(models[modelInd], nClasses, n_loops, batch_size, path)

            # print('group : ' + str(group) + 'subgroup ' + str(subgroup))

            # print('ypred_test. shape : ',ypred_test.shape)

            if group == 0:

                for i,elem in enumerate(ypred_test):

                    for bird_ind in indices_of_top_values(elem, 2):
                        predictionsGroup0[i].append(training_list[bird_ind])

                    # for classe,soft in enumerate(elem):
                    #     if (1 - soft) <= q_hat:
                    #         predictionsGroup0[i].append(training_list[classe])

            else:

                for i,elem in enumerate(ypred_test):

                    for bird_ind in indices_of_top_values(elem, 2):
                        bird = training_list[bird_ind]
                        if bird in predictionsGroup0[i]:
                            predictions[i].append(bird)

                    # for classe,soft in enumerate(elem):
                    #     bird = training_list[classe]
                    #     if (1 - soft) <= q_hat and bird in predictionsGroup0[i] :
                    #         predictions[i].append(bird)
    
    # print(predictions)

    for i in range(48):
        if predictions[i] != []:
            for species in predictions[i]:
                probabilities[competition_classes.index(species)] = 0.99999999
        row_id = file_id + str(5*(i+1))
        print(row_id)
        sample_sub.loc[sample_sub.row_id == row_id, competition_classes] = probabilities
        probabilities = np.ones((182,),dtype=np.float32)*4.798188e-08
    
    return sample_sub

def main():
    dataProcessor = melSpectrogram(seconds=5,sr=32000,n_mels=224,hop_length=716)
    models = loadModels()
    groups = extractSubgroups()
    # train_metadata = pd.read_csv("/kaggle/input/birdclef-2023/train_metadata.csv")
    train_metadata = pd.read_csv("C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/train_metadata.csv")
    competition_classes = sorted(train_metadata.primary_label.unique())
    # test_samples = list(glob.glob("/kaggle/input/birdclef-2023/test_soundscapes/*.ogg"))
    test_samples = list(glob.glob("C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/unlabeled_soundscapes_test/"))
    # sample_sub = pd.read_csv("/kaggle/input/birdclef-2023/sample_submission.csv")
    sample_sub = pd.read_csv("C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/sample_submission.csv")
    sample_sub[competition_classes] = sample_sub[competition_classes].astype(np.float32)
    
    for sample_filename in test_samples:
        sample_sub = predict_for_sample(sample_filename, sample_sub, dataProcessor, models, groups, competition_classes)

    sample_sub.to_csv("submission.csv", index=False)

    # sample_sub.to_csv("submission.csv", index=False)

if __name__ == '__main__': 
    main()




