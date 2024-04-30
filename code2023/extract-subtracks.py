import pandas as pd
import os
from pathlib import Path
import torchaudio
import h5py
import torch
torch.backends.cudnn.benchmark = True

transform = torchaudio.transforms.Resample(32000, 16000)
path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/train_audio/'
# ms_per_subtrack = 2000
# steps_per_subtrack = (ms_per_subtrack//20)
steps_per_subtrack = 32000
principal_only = True
train_metadata = pd.read_csv('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/train_metadata.csv')

species_dict = {}
size_dict = {}
pathlist = Path(r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\train_audio').glob('*')
i = 0
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    species = path_in_str[60:]
    species_dict[species] = i
    s = 0
    for ele in os.scandir(path_in_str):
        s += os.path.getsize(ele)
    size_dict[species] = s
    i += 1
sorted_size_dict = sorted(size_dict.items(), key=lambda x:x[1], reverse=True)
converted_dict = dict(sorted_size_dict)

# print(converted_dict)

# print(converted_dict.keys())

sorted_list = list(converted_dict.keys())

batch1 = {}
batch2 = {}
for j in range(11):
    batch2['list_' + str(2*j)] = [sorted_list[12*2*j] ,sorted_list[12*2*j + 2] ,sorted_list[12*2*j + 4] ,sorted_list[12*2*j +6] ,
                                  sorted_list[12*2*j + 8] ,sorted_list[12*2*j + 10] ,sorted_list[12*2*j + 12] ,sorted_list[12*2*j + 14] ,
                                   sorted_list[12*2*j + 16] ,sorted_list[12*2*j + 18] ,sorted_list[12*2*j + 20] ,sorted_list[12*2*j + 22]  ]
    
    batch2['list_' + str(2*j + 1)] =  [sorted_list[12*2*j + 1] ,sorted_list[12*2*j + 3] ,sorted_list[12*j + 5] ,sorted_list[12*2*j +7] ,
                                  sorted_list[12*2*j + 9] ,sorted_list[12*2*j + 11] ,sorted_list[12*2*j + 13] ,sorted_list[12*2*j + 15] ,
                                   sorted_list[12*2*j + 17] ,sorted_list[12*2*j + 19] ,sorted_list[12*2*j + 21] ,sorted_list[12*2*j + 23]  ]
    
    batch1['list_' + str(2*j)] = [sorted_list[12*j] ,sorted_list[12*j + 1] ,sorted_list[12*j + 2] ,sorted_list[12*j +3] ,
                                  sorted_list[12*j + 4] ,sorted_list[12*j + 5] ,sorted_list[12*j + 6] ,sorted_list[12*j + 7] ,
                                   sorted_list[12*j + 8] ,sorted_list[12*j + 9] ,sorted_list[12*j + 10] ,sorted_list[12*j + 11]  ]
    
    batch1['list_' + str(2*j + 1)] = [sorted_list[12*(j+1)] ,sorted_list[12*(j+1) + 1] ,sorted_list[12*(j+1) + 2] ,sorted_list[12*(j+1) +3] ,
                                  sorted_list[12*(j+1) + 4] ,sorted_list[12*(j+1) + 5] ,sorted_list[12*(j+1) + 6] ,sorted_list[12*(j+1) + 7] ,
                                   sorted_list[12*(j+1) + 8] ,sorted_list[12*(j+1) + 9] ,sorted_list[12*(j+1) + 10] ,sorted_list[12*(j+1) + 11]  ]
# print(batch1)
# print(batch2)

batch = batch1
n_list = 4
training_list = batch['list_' + str(n_list)]
species_dict = {}
for i,species in enumerate(training_list):
    species_dict[species] = i

# print(species_dict)

def select_training_list(species,training_list,rating):
    return species in training_list and rating > 4

train_metadata.insert(column='selected',loc=1,value=int)
for x in train_metadata.itertuples():
    train_metadata.at[x.Index,'selected'] = select_training_list(x.primary_label,species_dict,x.rating)

train = train_metadata[train_metadata['selected'] == 1].sample(frac=0.7,random_state=0)
test_calib = train_metadata[train_metadata['selected'] == 1].drop(train.index)
test = test_calib.sample(frac=0.5,random_state=0)
calib = test_calib.drop(test.index)

train_metadata.insert(column='ttc',loc=2,value=str)
for x in train_metadata.itertuples():
    if x.Index in train.index:
        train_metadata.at[x.Index,'tort'] = 'train'
    elif x.Index in test.index:
        train_metadata.at[x.Index,'tort'] = 'test'
    elif x.Index in calib.index:
        train_metadata.at[x.Index,'tort'] = 'calib'

k = 0

path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/train_audio/'

for x in train_metadata[train_metadata['selected'] == 1].itertuples():
    pl = x.primary_label
    sl = x.secondary_labels
    rating = str(x.rating)
    type = x.type
    audio_index = str(x.Index)
    # print(index)
    id = x.filename
    file = path + id
    data, samplerate = torchaudio.load(file, normalize=True)
    # print('data before')
    # print(data.shape)
    # print(type(data))
    # print(data.shape)
    # data = transform(data)
    # print(type(data))
    # print(data.shape)
    # data = data.cpu().detach().numpy()
    data = transform(data)
    # print('data')
    # print(data.shape)
    track_length = data.shape[1]
    tl = str(track_length)
    # print(type(data))
    # print(data.shape)

    # with torch.no_grad():
    #   input_values = processor(data, return_tensors="pt",sampling_rate = 16000).input_values  # Batch size 1
    #   # print('ok')
    #   # print(input_values.squeeze(1).shape)
    #   hidden_states = model(input_values.squeeze(1)).last_hidden_state.detach().numpy()
    # print('done')
    # track_length = hidden_states.shape[1]


    # dictionary = {'track_id' : id,
    #     'track_length' : track_length,
    #     'primary_label' : x.primary_label ,
    #     'secondary_labels' : x.secondary_labels ,
    #     'type' : x.type ,
    #     'latitude' : x.latitude,
    #     'longitude' : x.longitude,
    #     'rating' : x.rating,
    #     # 'x' : hidden_states[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack,:],
    #     # 'y_primary' : to_categorical(species_dict[x.primary_label],num_classes = 264),
    #     # 'y_all' : a,
    #     }
    
    if x.tort == 'train':
        for i in range(track_length // steps_per_subtrack):
            subtrack_index = str(k)
            torch.save(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], 'data/subtracks64000/train/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
            k += 1
    elif x.tort == 'test':
        for i in range(track_length // steps_per_subtrack):
            subtrack_index = str(k)
            torch.save(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], 'data/subtracks64000/test/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
            k += 1
    elif x.tort == 'calib':
        for i in range(track_length // steps_per_subtrack):
            subtrack_index = str(k)
            torch.save(data[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack], 'data/subtracks64000/calib/' + subtrack_index + '_' + audio_index  + '_' + pl + '_' + sl + '_' + tl + '_'  + '_' + rating +  '.pt')
            k += 1


        # print(type(df['x'][0])) 
        # print(df.head())
        
# df.to_csv('2000.csv')


