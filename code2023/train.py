import torch.nn as nn
from torch import optim
import pandas as pd
import torchaudio
import torch
torch.backends.cudnn.benchmark = True

from datasets import load_dataset
from pathlib import Path
import re
import json
from transformers import AutoConfig , AutoTokenizer , AutoFeatureExtractor , Wav2Vec2Model , Wav2Vec2Processor
torch.cuda.empty_cache()
import time

model_checkpoint = "facebook/wav2vec2-large-xlsr-53" # 'facebook/wav2vec2-base-960h' #"facebook/wav2vec2-large-xlsr-53"

common_voice_train = load_dataset("common_voice", "tr", split="train+validation")
common_voice_test = load_dataset("common_voice", "tr", split="test")

common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

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

with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

config = AutoConfig.from_pretrained(model_checkpoint)

tokenizer_type = config.model_type if config.tokenizer_class is None else None
config = config if config.tokenizer_class is not None else None


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

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

del feature_extractor
del tokenizer

wav2vec = Wav2Vec2Model.from_pretrained(model_checkpoint).to('cuda:0')

class LSTM(nn.Module):
    """
    input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediciton_periods input to get_x_y_pairs
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)


        
        self.linear = nn.Linear(hidden_size, output_size)
        # self.linear2 = nn.Linear(output_size, 264)

        # self.soft = nn.Softmax(dim = 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, batch_size , hidden=None):
        if hidden==None:
            self.hidden = (torch.zeros(1,batch_size,self.hidden_size).to('cuda:0'),
                           torch.zeros(1,batch_size,self.hidden_size).to('cuda:0'))
        else:
            self.hidden = hidden
            
        """
        inputs need to be in the right shape as defined in documentation
        - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        
        lstm_out - will contain the hidden states from all times in the sequence
        self.hidden - will contain the current hidden state and cell state
        """
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # lstm_out, self.hidden = self.lstm(x.view(len(x),1,-1), 
        #                                   self.hidden)
        # print('ee')
        # print(lstm_out.shape)
        # print('lstm_out')
        # print(lstm_out.shape)
        predictions = self.linear(lstm_out)
        # print('pred shape')
        # print(predictions.shape)

        # predictions = self.relu(predictions)
        # print('pred ras nami')
        # print(predictions[:,-1,:].shape)

        # predictions = self.linear2(predictions)

        # predictions = self.soft(predictions)

        # preds = self.soft(predictions[:,-1,:])
        # print(predictions4[:,-1,:])
        
        return predictions[:,-1,:], self.hidden
    

species_dict = {}
pathlist = Path(r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\train_audio').glob('*')
i = 0
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    species = path_in_str[60:]
    species_dict[species] = i
    i += 1
# print(species_dict)
    


train_metadata = pd.read_csv('C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/train_metadata.csv')
path = 'C:/Users/fares/OneDrive/Bureau/kaggleBirds/data/train_audio/'


def select_group(species,dic,n):
    return dic[species] // n

train_metadata.insert(column='group',loc=1,value=int)
for x in train_metadata.itertuples():
    train_metadata.at[x.Index,'group'] = select_group(x.primary_label,species_dict,10)

transform = torchaudio.transforms.Resample(32000, 16000)
steps_per_subtrack = 64000
    
model = LSTM(input_size=1024, hidden_size=100, output_size=10).to('cuda:0') #######
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train = train_metadata[train_metadata['group'] == 0].sample(frac=0.8,random_state=0)
test = train_metadata[train_metadata['group'] == 0].drop(train.index)
epochs = 600
batch_size = 4
model.train()
for epoch in range(epochs):
    start = time.time()
    # shuffled = train_metadata.sample(frac=1).reset_index()
    # for x,y in zip(x_train, y_train):
    for k in train.itertuples():
        train = train.sample(frac=1,random_state=0)
        file = path + k.filename
        data, samplerate = torchaudio.load(file, normalize=True)
        data = transform(data)
        track_length = data.shape[1]
        if (track_length > steps_per_subtrack):
            
            nelem = (track_length // steps_per_subtrack)
            if (nelem <= batch_size ):
                t = data[:,:steps_per_subtrack]
                for i in range(nelem - 1):
                    t = torch.cat((t,data[:,(i)*steps_per_subtrack:(i+1)*steps_per_subtrack]),dim=0)
                with torch.no_grad():
                    input_values = processor(t, return_tensors="pt",sampling_rate = 16000).input_values.to('cuda:0')  # Batch size 1
                    # print(input_values.shape)
                    # print('input values')
                    # print(input_values.device)
                    # print('ok')
                    # print(input_values.squeeze(0).shape)
                    hidden_states = wav2vec(input_values.squeeze(0)).last_hidden_state
                    input_values.detach()
                    # print(hidden_states.shape)
                hidden_states.requires_grad_()
                size = hidden_states.shape[0]
                # print(hidden_states.shape)
                # print(nelem)
                y_hat, _ = model(hidden_states, nelem , None)
                # print(y_hat.get_device())
                y_hat.requires_grad_()
                # y_hat, _ = model(hidden_states.view((199,size,1024)), None) 
                # y_hat, _ = model(torch.reshape(hidden_states,(199,size,1024)), None) 
                # y = to_categorical(species_dict[x.primary_label],num_classes = 9)
                # y = torch.as_tensor(species_dict[k.primary_label])
                y = torch.ones(size,device='cuda:0',dtype=torch.long)*species_dict[k.primary_label]
                # print(y.get_device())
                # y.requires_grad_()
                # print(y.shape)
                optimizer.zero_grad()
                # print(y_hat.shape)
                loss = criterion(y_hat, y)
                loss.backward()
                # for param in model.parameters():
                #     print(param.shape)
                #     print(param.grad is None)
                optimizer.step()
            else:
                n_loops = nelem // batch_size
                reste = nelem % batch_size
                for j in range(n_loops):
                    t = data[:,batch_size*j*steps_per_subtrack:(batch_size*j+1)*steps_per_subtrack]
                    for i in range(batch_size - 1):
                        t = torch.cat((t,data[:,(batch_size*j+i)*steps_per_subtrack:(batch_size*j+i+1)*steps_per_subtrack]),dim=0)
                    with torch.no_grad():
                        input_values = processor(t, return_tensors="pt",sampling_rate = 16000).input_values.to('cuda:0')  # Batch size 1
                        # print(input_values.shape)
                        # print('input values')
                        # print(input_values.device)
                        # print('ok')
                        # print(input_values.squeeze(0).shape)
                        hidden_states = wav2vec(input_values.squeeze(0)).last_hidden_state
                        input_values.detach()
                        # print(hidden_states.shape)
                    hidden_states.requires_grad_()
                    size = hidden_states.shape[0]
                    # print(hidden_states.shape)
                    # print(batch_size)
                    y_hat, _ = model(hidden_states, batch_size , None)
                    # print(y_hat.get_device())
                    y_hat.requires_grad_()
                    # y_hat, _ = model(hidden_states.view((199,size,1024)), None) 
                    # y_hat, _ = model(torch.reshape(hidden_states,(199,size,1024)), None) 
                    # y = to_categorical(species_dict[x.primary_label],num_classes = 9)
                    # y = torch.as_tensor(species_dict[k.primary_label])
                    y = torch.ones(size,device='cuda:0',dtype=torch.long)*species_dict[k.primary_label]
                    # print(y.get_device())
                    # y.requires_grad_()
                    # print(y.shape)
                    optimizer.zero_grad()
                    # print(y_hat.shape)
                    loss = criterion(y_hat, y)
                    loss.backward()
                    # for param in model.parameters():
                    #     print(param.grad)
                    optimizer.step()
                if reste != 0:
                    done = batch_size * n_loops
                    t = data[:,done*steps_per_subtrack:(done + 1)*steps_per_subtrack]
                    for i in range(reste - 1):
                        t = torch.cat((t,data[:,(done+i+1)*steps_per_subtrack:(done+i+2)*steps_per_subtrack]),dim=0)
                    with torch.no_grad():
                        input_values = processor(t, return_tensors="pt",sampling_rate = 16000).input_values.to('cuda:0')  # Batch size 1
                        # print(input_values.shape)
                        # print('input values')
                        # print(input_values.device)
                        # print('ok')
                        # print(input_values.squeeze(0).shape)
                        hidden_states = wav2vec(input_values.squeeze(0)).last_hidden_state
                        input_values.detach()
                        # print(hidden_states.shape)
                    hidden_states.requires_grad_()
                    size = hidden_states.shape[0]
                    y_hat, _ = model(hidden_states, reste , None)
                    # print(y_hat.get_device())
                    y_hat.requires_grad_()
                    # y_hat, _ = model(hidden_states.view((199,size,1024)), None) 
                    # y_hat, _ = model(torch.reshape(hidden_states,(199,size,1024)), None) 
                    # y = to_categorical(species_dict[x.primary_label],num_classes = 9)
                    # y = torch.as_tensor(species_dict[k.primary_label])
                    y = torch.ones(size,device='cuda:0',dtype=torch.long)*species_dict[k.primary_label]
                    # print(y.get_device())
                    # y.requires_grad_()
                    # print(y.shape)
                    optimizer.zero_grad()
                    # print(y_hat.shape)
                    loss = criterion(y_hat, y)
                    loss.backward()
                    # for param in model.parameters():
                    #     print(param.grad)
                    optimizer.step()


        # else:
            # print('7achya')
        
        
    if epoch%1==0:
        end = time.time()
        print('time : ',end - start)
        # print(end - start)
        print(f'epoch: {epoch:4} loss:{loss.item():10.8f}')
    if epoch != 0 and epoch%5 == 0:
        torch.save(model.state_dict(), "C:/Users/fares/OneDrive/Bureau/kaggleBirds/model0")