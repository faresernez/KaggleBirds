import pandas as pd
import os
# from keras.utils.np_utils import to_categorical
import numpy as np
from pathlib import Path
import torchaudio
import h5py
import torch
torch.backends.cudnn.benchmark = True
# import tensorflow as tf
# tf.autograph.set_verbosity(0)
import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# tf.compat.v1.disable_eager_execution()



model_checkpoint =   "facebook/wav2vec2-large-xlsr-53" #'facebook/wav2vec2-base-960h'
batch_size = 16

from datasets import load_dataset, load_metric, Audio

common_voice_train = load_dataset("common_voice", "tr", split="train+validation")
common_voice_test = load_dataset("common_voice", "tr", split="test")

common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

import re
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = common_voice_train.map(
  extract_all_chars, batched=True, 
  batch_size=-1, keep_in_memory=True, 
  remove_columns=common_voice_train.column_names
)
vocab_test = common_voice_test.map(
  extract_all_chars, batched=True, 
  batch_size=-1, keep_in_memory=True, 
  remove_columns=common_voice_test.column_names
)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_checkpoint)

tokenizer_type = config.model_type if config.tokenizer_class is None else None
config = config if config.tokenizer_class is not None else None

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
  "./",
  config=config,
  tokenizer_type=tokenizer_type,
  unk_token="[UNK]",
  pad_token="[PAD]",
  word_delimiter_token="|",
)

del vocab_list
del vocab_train
del vocab_test
del common_voice_train
del common_voice_test



# from transformers import AutoFeatureExtractor, AutoModelForCTC

# feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

# from transformers import Wav2Vec2Processor

# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

from transformers import AutoFeatureExtractor, AutoModelForCTC , Wav2Vec2Model
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

from transformers import Wav2Vec2Processor,  AutoModelForPreTraining, TFWav2Vec2Model


processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

del feature_extractor
del tokenizer

# model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53",from_pt=True)
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

species_dict = {}
pathlist = Path(r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\train_audio').glob('*')
i = 0
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    species = path_in_str[60:]
    # if species == 'abythr1' or species == 'afbfly1' or species == 'afdfly1' or species == 'affeag1' or species == 'afgfly1' or species == 'afmdov1' or species == 'afpkin1' or species == 'afrgrp1' or species == 'afrjac1':
    species_dict[species] = i
    i += 1

sp = ''
transform = torchaudio.transforms.Resample(32000, 16000)
path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/train_audio/'
ms_per_subtrack = 4000
steps_per_subtrack = (ms_per_subtrack//20)
# principal_only = True
# df = pd.DataFrame(columns=['track_id','track_length','primary_label','secondary_labels' ,'type','latitude','longitude','rating','x','y_primary','y_all'])
train_metadata = pd.read_csv('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/train_metadata.csv')
# df.insert(column='num_tokens',loc=13,value=int)
# a = np.zeros((264,))
k = 2128
# while (True):
for x in train_metadata.itertuples():
    print(x.Index)
    pl = x.primary_label
    sl = x.secondary_labels
    # print(type(pl))
    # print(type(sl))
    # if x.primary_label != sp:
    #     sp = x.primary_label
    #     print(sp)
        # if sp == 'afrthr1':
        #     break
    # url = x.url
    # print(url)
    # if  sp != 'abethr1' and sp != 'abhori1' and sp != 'afecuc1' and sp != 'afghor1' and sp != 'afpfly1' and sp != 'afpwag1' and sp != 'afrgos1':
    index = str(x.Index)
    # print(index)
    id = x.filename
    # print(id)
    file = path + id
    
    # print('data')
    # print(data.shape)
    # print(type(data))
    # print('1')
    # print(type(data))
    # print(np.squeeze(data).shape)
    # print(np.squeeze(data))
    # print('1')
    # data = data.detach().cpu().numpy()[0]
    # print('2')
    # print(data.shape)
    # print('2')
    # data = np.asarray(data)
    # print('3')
    # print(data.shape)
    # print('3')
    # print(type(data))
    # print(data)
    # print('4')
    # print(isinstance(data, (np.ndarray, np.generic)))
    # print('start')
    with torch.no_grad():
      data, samplerate = torchaudio.load(file, normalize=True)
      # print(type(data))
      # print(data.shape)
      data = transform(data)
      input_values = processor(data, return_tensors="pt",sampling_rate = 16000).input_values  # Batch size 1
      # print('ok')
      # print(input_values.squeeze(1).shape)
      hidden_states = model(input_values.squeeze(1)).last_hidden_state.detach().numpy()
    # print('hidden')
    # print(hidden_states.shape)
    # print('done')
    track_length = hidden_states.shape[1]


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
    
    for i in range(track_length // steps_per_subtrack):
        # print(k)
        if (k % 1000 == 0):
            print(k)
            # df.to_csv('2000.csv')

        # l = x.secondary_labels
        # print(l)
        # print(type(l))
        # for label in l:
        #     print(label)
        #     a = a + to_categorical(species_dict[label],num_classes = 264),

        # dictionary = {'track_id' : id,
        # 'track_length' : track_length,
        # 'primary_label' : x.primary_label ,
        # 'secondary_labels' : x.secondary_labels ,
        # 'type' : x.type ,
        # 'latitude' : x.latitude,
        # 'longitude' : x.longitude,
        # 'rating' : x.rating,
        # 'x' : hidden_states[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack,:],
        # 'y_primary' : to_categorical(species_dict[x.primary_label],num_classes = 264),
        # 'y_all' : a,
        # 
        # 
        # }

        # dictionary['x'] = hidden_states[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack,:]
        group = str(k)
        torch.save(hidden_states[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack,:], 'data/2000_new/' + group + '_' + index  + '_' + pl + '_' + sl + '.pt')
        # print('stqrt enre')
        # with h5py.File('data/2000_new/' + group + '_' + index + '.hdf5', 'w') as f:
        #     # f.create_group(group)
        #     f.create_dataset(group, data=hidden_states[:,i*steps_per_subtrack:(i+1)*steps_per_subtrack,:])
        # print('ok')
        # print('fin enre')

        # output: = pd.DataFrame()


        # df_dictionary = pd.DataFrame([dictionary])
        # df = pd.concat([df, df_dictionary], ignore_index=True)
        k += 1


        # print(type(df['x'][0])) 
        # print(df.head())
        
# df.to_csv('2000.csv')

