import shutil
import os
import librosa
import numpy as np
from pathlib import Path
import random
import torch
import time
from concurrent.futures import ThreadPoolExecutor

def audio_to_chunks(audio_file,steps_per_subtrack = 160000, sr=32000):
    chunks = []
    data, samplerate = librosa.load(audio_file, sr=sr)
    track_length = data.shape[0]
    nChunks = track_length // steps_per_subtrack
    if (nChunks == 0): #if an audio is shorter than steps_per_subtrack, we duplicate it
        while (data.shape[0] < steps_per_subtrack):
            data = np.tile(data,2)
        nChunks = 1
    for i in range(nChunks):
        chunks.append(data[i*steps_per_subtrack:(i+1)*steps_per_subtrack])
    chunks.append(data[-steps_per_subtrack:]) #adding the last steps_per_subtrack of the audio to not miss anything
    return chunks , samplerate

def chunk_to_spectrum(chunk,samplerate,n_mels=224, hop_length=716, fmax=16000): # S.shape = (X,224,313)
    S = librosa.feature.melspectrogram(y=chunk, sr=samplerate, n_mels=n_mels, hop_length=hop_length, fmax=fmax)
    S = librosa.util.normalize(S)
    # mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S))
    return S

def extractAllSpecies():
    dataPath = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/BirdClef2024/train_audio/'
    x = len(dataPath)
    species_dict = {}
    size_dict = {}
    i = 0
    for path in Path(dataPath).glob('*'):
        # because path is object not string
        path_in_str = str(path)
        species = path_in_str[x:]
        species_dict[species] = i
        s = 0
        for ele in os.scandir(path_in_str):
            s += os.path.getsize(ele)
        size_dict[species] = s/(1024*1024)
        i += 1
    sorted_size_dict = sorted(size_dict.items(), key=lambda x:x[1], reverse=True)
    converted_dict = dict(sorted_size_dict)
    return list(converted_dict.keys())

def extractSubgroups():
    groups = []
    subgroups = []
    sorted_list = extractAllSpecies()
    bird0 = sorted_list[0]
    bird1 = sorted_list[1]
    sorted_list = sorted_list[2:]
    for j in range(12):
        l = [sorted_list[15*j] ,sorted_list[15*j + 1] ,sorted_list[15*j + 2] ,sorted_list[15*j +3] ,
            sorted_list[15*j + 4] ,sorted_list[15*j + 5] ,sorted_list[15*j + 6] ,sorted_list[15*j + 7] ,
            sorted_list[15*j + 8] ,sorted_list[15*j + 9] ,sorted_list[15*j + 10] ,sorted_list[15*j + 11] ,
            sorted_list[15*j + 12] ,sorted_list[15*j + 13] ,sorted_list[15*j + 14]]
        subgroups.append(l)
    groups.append(subgroups)
    subgroups = []
    for j in range(6):
        l = [sorted_list[15*2*j] ,sorted_list[15*2*j + 2] ,sorted_list[15*2*j + 4] ,sorted_list[15*2*j +6] ,
            sorted_list[15*2*j + 8] ,sorted_list[15*2*j + 10] ,sorted_list[15*2*j + 12] ,sorted_list[15*2*j + 14] ,
            sorted_list[15*2*j + 16] ,sorted_list[15*2*j + 18] ,sorted_list[15*2*j + 20] ,sorted_list[15*2*j + 22] ,
            sorted_list[15*2*j + 24] ,sorted_list[15*2*j + 26] ,sorted_list[15*2*j + 28]]
        subgroups.append(l)
        l = [sorted_list[15*2*j + 1] ,sorted_list[15*2*j + 3] ,sorted_list[15*j + 5] ,sorted_list[15*2*j +7] ,
            sorted_list[15*2*j + 9] ,sorted_list[15*2*j + 11] ,sorted_list[15*2*j + 13] ,sorted_list[15*2*j + 15] ,
            sorted_list[15*2*j + 17] ,sorted_list[15*2*j + 19] ,sorted_list[15*2*j + 21] ,sorted_list[15*2*j + 23] ,
            sorted_list[15*2*j + 25] ,sorted_list[15*2*j + 27] ,sorted_list[15*j + 29]]
        subgroups.append(l)
    groups.append(subgroups)
    groups[0][0].append(bird0)
    groups[0][0].append(bird1)
    groups[1][0].append(bird0)
    groups[1][1].append(bird1)
    return groups

def extract(dataProcessor,ratioTrainTestCalib,dataPath,destination,classes,extractionDone,byFileExtraction=True):
        if (not extractionDone):
        
            folders = [ 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/Birdclef2024/finetuning/train/',
                        'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/Birdclef2024/finetuning/test/',
                        'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/Birdclef2024/finetuning/calib/',]
            for folder in folders:
                shutil.rmtree(folder)
                os.mkdir(folder)

            BirdClassMap = {}
            filesMap = {}
            classMap = {}
            ind = 0 
            indDict = {'train/':[],'test/':[],'calib/':[]}
            for i,c in enumerate(classes):
                stri = str(i)
                BirdClassMap[c] = i
                classPath = dataPath + c + '/'
                for path in Path(classPath).glob('*'):
                    chunks = dataProcessor.loadAudio(path)

                    if (byFileExtraction):
                        folder = trainTestCalib(ratioTrainTestCalib)
                        for chunk in chunks:
                            indDict, filesMap, classMap, ind = saveChunk(chunk,i,destination,folder,ind,stri,dataProcessor,indDict,filesMap,classMap)
                    else:
                        for chunk in chunks:
                            folder = trainTestCalib(ratioTrainTestCalib)
                            indDict, filesMap, classMap, ind = saveChunk(chunk,i,destination,folder,ind,stri,dataProcessor,indDict,filesMap,classMap)

            return BirdClassMap , indDict['train/'] , indDict['test/'] , indDict['calib/'] , filesMap , classMap
        
        else:

            BirdClassMap = {}
            filesMap = {}
            classMap = {}
            ind = 0 
            indDict = {'train/':[],'test/':[],'calib/':[]}
            for i,c in enumerate(classes):
                stri = str(i)
                BirdClassMap[c] = i
                classPath = dataPath + c + '/'
                for path in Path(classPath).glob('*'):
                    chunks = dataProcessor.loadAudio(path)
                    if (byFileExtraction):
                        folder = trainTestCalib(ratioTrainTestCalib)
                        for k in range(len(chunks)):
                            indDict, filesMap, classMap, ind = saveChunkWithoutExtraction(i,destination,folder,ind,stri,dataProcessor,indDict,filesMap,classMap)
                    else:
                        for k in range(len(chunks)):
                            folder = trainTestCalib(ratioTrainTestCalib)
                            indDict, filesMap, classMap, ind = saveChunkWithoutExtraction(i,destination,folder,ind,stri,dataProcessor,indDict,filesMap,classMap)

            return BirdClassMap , indDict['train/'] , indDict['test/'] , indDict['calib/'] , filesMap , classMap
        
def saveChunk(chunk,i,destination,folder,ind,stri,dataProcessor,indDict,filesMap,classMap):
    tensorPath = destination + folder + str(ind) + '_' + stri + '.pt'
    torch.save(torch.from_numpy(dataProcessor.processChunk(chunk)),tensorPath)
    indDict[folder].append(ind)
    filesMap[ind] = tensorPath
    classMap[ind] = i
    ind += 1
    return indDict, filesMap, classMap, ind

def saveChunkWithoutExtraction(i,destination,folder,ind,stri,dataProcessor,indDict,filesMap,classMap):
    tensorPath = destination + folder + str(ind) + '_' + stri + '.pt'
    indDict[folder].append(ind)
    filesMap[ind] = tensorPath
    classMap[ind] = i
    ind += 1
    return indDict, filesMap, classMap, ind

def computeTrainingWeights(nClasses):
    w = np.zeros(nClasses)
    for path in Path(r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\Birdclef2024\finetuning\train').glob('*'):
        w[int(str(path).split('_')[1][:-3])] += 1
    w = w/np.sum(w)
    w = [1/ele for ele in w]
    return torch.FloatTensor(w)

def trainTestCalib(percentages = [0.8,0.2,0.]):
        r = random.random()
        if (percentages[0] == 1):
            return 'train/'
        elif (percentages[2] == 0 and r > percentages[1]):
            return 'train/'
        elif (percentages[2] == 0):
            return 'test/'
        elif (r < percentages[1]):
            return 1
        elif (r > 1 - percentages[2]):
            return 'calib/'
        else:
            return 'train/'

def extractForPyDataLoader(dataProcessor,ratioTrainTestCalib,dataPath,classes,extractionDone):

    folder = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/Birdclef2024/finetuning/'
    # BirdClassMap = {}
    partition = {'train':[],'test':[],'calib':[]}
    labels = {}

    if (not extractionDone):
        s = time.time()
        shutil.rmtree(folder)
        os.mkdir(folder)
        ind = 0
        # chunkId = 'id'
        for i,c in enumerate(classes):
            stri = str(i)
            # BirdClassMap[c] = i
            classPath = dataPath + c + '/'
            # print(classPath)
            for path in Path(classPath).glob('*'):
                chunks = dataProcessor.loadAudio(path)
                for chunk in chunks:
                    type = trainTestCalibForPyDataLoader(ratioTrainTestCalib)
                    partitionId = 'id-' + str(ind) + '_' + type + '_' + stri
                    labels[partitionId] = i
                    partition[type].append(partitionId)
                    destination = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/Birdclef2024/finetuning/' + partitionId  + '.pt'
                    torch.save(torch.from_numpy(dataProcessor.processChunk(chunk)).unsqueeze(0),destination)
                    ind += 1
                # saveChunks(chunks,destination,dataProcessor,tensorShape,dtype)
        e = time.time()
        print('time extracting : ',e-s)
    else:   

        for path in Path(folder).glob('*'):

            a = str(path).split('id-')[1]
            b = a.split('_')
            type = b[1]
            stri = b[2][:-3]
            partitionId = 'id-' + b[0] + '_' + type + '_' + stri
            label = int(stri)

            partition[type].append(partitionId)
            labels[partitionId] = label

    return partition, labels

def saveChunksForPyDataLoader(chunks,destination,dataProcessor,tensorShape,dtype):
    batch = torch.empty((0,1) + tensorShape, dtype = dtype)
    nChunks = len(chunks)
    for chunk in chunks:
        batch = torch.cat((batch,torch.from_numpy(dataProcessor.processChunk(chunk)).unsqueeze(0).unsqueeze(0)),dim = 0)
    torch.save(batch,destination)

def computeTrainingWeightsForPyDataLoader(nClasses):
    w = np.zeros(nClasses)
    for path in Path(r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\Birdclef2024\finetuning').glob('*'):
        a = str(path).split('_')
        if a[1] == 'train':
            w[int(a[2][:-3])] += 1
    w = w/np.sum(w)
    w = [1/ele for ele in w]
    return torch.FloatTensor(w)

def trainTestCalibForPyDataLoader(percentages = [0.8,0.2,0.]):
        r = random.random()
        if (percentages[0] == 1):
            return 'train'
        elif (percentages[2] == 0 and r > percentages[1]):
            return 'train'
        elif (percentages[2] == 0):
            return 'test'
        elif (r < percentages[1]):
            return 1
        elif (r > 1 - percentages[2]):
            return 'calib'
        else:
            return 'train'

def extractWithWorkers(dataProcessor, ratioTrainTestCalib, dataPath, classes, extractionDone, num_workers): #extraction with multiprocessing only implemented for PyTorch DataLoader
    folder = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/Birdclef2024/finetuning/'
    partition = {'train':[], 'test':[], 'calib':[]}
    labels = {}

    if not extractionDone:
        s = time.time()
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        ind = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i, c in enumerate(classes):
                classPath = dataPath + c + '/'
                for path in Path(classPath).glob('*'):
                    futures.append(executor.submit(saveChunkWithWorkers, dataProcessor, i, path, ind, ratioTrainTestCalib, folder, labels, partition))
            
            # Handling exceptions and collecting results
            for future in futures:
                try:
                    future.result()  # Ensure all futures complete successfully
                except Exception as e:
                    print(f"An error occurred in the threading process: {e}")
        
        print('Time extracting: ', time.time() - s)
    else:
        for path in Path(folder).glob('*'):
            a = str(path).split('id-')[1]
            b = a.split('_')
            type = b[1]
            stri = b[2][:-3]
            partitionId = 'id-' + b[0] + '_' + type + '_' + stri
            label = int(stri)
            partition[type].append(partitionId)
            labels[partitionId] = label

    return partition, labels

def saveChunkWithWorkers(dataProcessor, i, path, ind, ratioTrainTestCalib, folder, labels, partition):
    try:
        chunks = dataProcessor.loadAudio(path)
        results = []
        type = trainTestCalibForPyDataLoader(ratioTrainTestCalib)
        for chunk in chunks:
            partitionId = 'id-' + str(ind) + '_' + type + '_' + str(i)
            labels[partitionId] = i
            partition[type].append(partitionId)
            destination = folder + partitionId + '.pt'
            torch.save(torch.from_numpy(dataProcessor.processChunk(chunk)).unsqueeze(0), destination)
            ind += 1
            results.append((partitionId, destination))
        return results
    except Exception as e:
        print(f"Error processing file {path}: {e}")
        return []
    
def indices_of_top_values(values, num_top):
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return sorted_indices[:num_top]





### TEST ###

# float_values = [3.5, 2.1, 4.7, 1.8, 4.7, 0.9]
# top_indices = indices_of_top_values(float_values, num_top=2)
# print(top_indices) 

groups = extractSubgroups()
print(groups)
# print('groups[0]')
# print(groups[0])
# print('groups[1]')
# print(groups[1])

# for j in range(2):
#     for i in range(12):
#         print('group : ' + str(j) + ' subgroup : ' + str(i) )
#         print(groups[j][i])


# audio_file = r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\Birdclef2024\train_audio\asiope1\XC397761.ogg' #hop length aug -> x dimniue
# chunks, samplerate = audio_to_chunks(audio_file)
# print(len(chunks))
# S = chunk_to_spectrum(chunks[0], samplerate=samplerate, hop_length=716)
# print(S.shape)

# computeTrainingWeights(12)