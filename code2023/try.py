import pandas as pd
import os
from pathlib import Path
import torchaudio
# import h5py
import torch
torch.backends.cudnn.benchmark = True
import pandas as pd
import os
import time
import numpy as np
from pathlib import Path
import torchaudio
# import h5py
import torch
torch.backends.cudnn.benchmark = True
import shutil
import random
import torch.nn as nn
from torch import optim
from code2023.calibration import calibrate


### WAV2VEC MODEL ###

# model_checkpoint =   'facebook/wav2vec2-base-960h' 
# from datasets import load_dataset, load_metric, Audio

# common_voice_train = load_dataset("common_voice", "tr", split="train+validation")
# common_voice_test = load_dataset("common_voice", "tr", split="test")

# common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
# common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

# import re
# chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

# def remove_special_characters(batch):
#     batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
#     return batch

# def extract_all_chars(batch):
#   all_text = " ".join(batch["sentence"])
#   vocab = list(set(all_text))
#   return {"vocab": [vocab], "all_text": [all_text]}

# vocab_train = common_voice_train.map(
#   extract_all_chars, batched=True, 
#   batch_size=-1, keep_in_memory=True, 
#   remove_columns=common_voice_train.column_names
# )
# vocab_test = common_voice_test.map(
#   extract_all_chars, batched=True, 
#   batch_size=-1, keep_in_memory=True, 
#   remove_columns=common_voice_test.column_names
# )

# vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

# vocab_dict = {v: k for k, v in enumerate(vocab_list)}
# vocab_dict

# vocab_dict["|"] = vocab_dict[" "]
# del vocab_dict[" "]

# vocab_dict["[UNK]"] = len(vocab_dict)
# vocab_dict["[PAD]"] = len(vocab_dict)
# len(vocab_dict)

# import json
# with open('vocab.json', 'w') as vocab_file:
#     json.dump(vocab_dict, vocab_file)



# del vocab_list
# del vocab_train
# del vocab_test
# del common_voice_train
# del common_voice_test

# from transformers import AutoFeatureExtractor, AutoModelForCTC , Wav2Vec2Model
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)




# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# # del feature_extractor
# # del tokenizer

# model = Wav2Vec2Model.from_pretrained(model_checkpoint).to('cuda:0')


# feature_extractor.save_pretrained('C:/Users/fares/OneDrive/Bureau/kaggleBirds/wav2vec/')
# a = tokenizer.save_pretrained('C:/Users/fares/OneDrive/Bureau/kaggleBirds/wav2vec/')
# print(a)
# model.save_pretrained('C:/Users/fares/OneDrive/Bureau/kaggleBirds/wav2vec/')

# tokenizer = 

from transformers import AutoConfig , Wav2Vec2Processor , Wav2Vec2FeatureExtractor , Wav2Vec2Model

config = AutoConfig.from_pretrained('C:/Users/fares/OneDrive/Bureau/kaggleBirds/wav2vec/')

tokenizer_type = config.model_type if config.tokenizer_class is None else None
config = config if config.tokenizer_class is not None else None

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
  'C:/Users/fares/OneDrive/Bureau/kaggleBirds/wav2vec/',
  config=config,
  tokenizer_type=tokenizer_type,
  unk_token="[UNK]",
  pad_token="[PAD]",
  word_delimiter_token="|",
)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("C:/Users/fares/OneDrive/Bureau/kaggleBirds/wav2vec/preprocessor_config.json")

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = Wav2Vec2Model.from_pretrained('C:/Users/fares/OneDrive/Bureau/kaggleBirds/wav2vec/').to('cuda:0')

