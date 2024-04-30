import pandas as pd
import os
import time
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

start = time.time()

batch_size = 32
path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/'
model_checkpoint =   'facebook/wav2vec2-base-960h'  # 'facebook/wav2vec2-base-960h' "facebook/wav2vec2-large-xlsr-53"

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
# model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53") model_checkpoint
model = Wav2Vec2Model.from_pretrained(model_checkpoint).to('cuda:0')

# species_dict = {}
# pathlist = Path(r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\train_audio').glob('*')
# i = 0
# for path in pathlist:
#     # because path is object not string
#     path_in_str = str(path)
#     species = path_in_str[60:]
#     # if species == 'abythr1' or species == 'afbfly1' or species == 'afdfly1' or species == 'affeag1' or species == 'afgfly1' or species == 'afmdov1' or species == 'afpkin1' or species == 'afrgrp1' or species == 'afrjac1':
#     species_dict[species] = i
#     i += 1
# directory = 
 
# iterate over files in
# that directory
# for i,filename in enumerate(os.listdir(r"C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\subtracks64000")):
#     # f = os.path.join(directory, filename)
#     # checking if it is a file
#     # if os.path.isfile(f):
#     #     print(f)
#     print(i)
#     print(filename)
saving_path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/hidden64000/'

# batch_size = 64
# n_subtracks = 8762
# n_loops = n_subtracks // batch_size
# reste = n_subtracks % batch_size

mode = 'train/'
path += mode
l = os.listdir('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/subtracks64000/' + mode )
saving_path += mode
n_subtracks = len(l)
n_loops = n_subtracks // batch_size
reste = n_subtracks % batch_size
# i = 0

print('model device')
print(model.device)

for k in range(n_loops):
    start = time.time()
    t = torch.load(path + l[0 + batch_size*k], map_location='cuda:0')
    for i in range(batch_size - 1):
        x = torch.load(path + l[i + 1 + batch_size*k], map_location='cuda:0')
        t = torch.cat((t,x),dim=0)
    # print('tensor device')
    # print(t.device)
    # # print(t.shape)
    # end1 = time.time()
    # print('loading')
    # print(end1 - start)

    # start = time.time()
    with torch.no_grad():
        input_values = processor(t, return_tensors="pt",sampling_rate = 16000).input_values.to('cuda:0')  # Batch size 1
        # print(input_values.shape)
        # print('input values')
        # print(input_values.device)

        # print('ok')
        # print(input_values.squeeze(0).shape)
        hidden_states = model(input_values.squeeze(0)).last_hidden_state
        # print(hidden_states.shape)
    # end2 = time.time()
    # print('traitement')
    # print(end2 - start)
    for t in range(batch_size):
        torch.save(hidden_states[t], saving_path + l[t + 32*k])
    # start = time.time()
    # torch.save(hidden_states, saving_path + l[32*k])
    # end = time.time()
    # print('saving')
    # print(end - start)


    

if reste != 0:
    done = batch_size * n_loops
    t = torch.load(path + l[done], map_location='cuda:0')
    for i in range(reste - 1):
        x = torch.load(path + l[i + 1 + done], map_location='cuda:0')
        t = torch.cat((t,x),dim=0)
    # print(t.shape)

    with torch.no_grad():
        input_values = processor(t, return_tensors="pt",sampling_rate = 16000).input_values.to('cuda:0')  # Batch size 1
        # print(input_values.shape)

        # print('ok')
        # print(input_values.squeeze(0).shape)
        hidden_states = model(input_values.squeeze(0)).last_hidden_state
        # print(hidden_states.shape)

    for t in range(reste):
        torch.save(hidden_states[t], saving_path + l[t + done])
    # torch.save(hidden_states, saving_path + l[done])


end = time.time()
print(end - start)
