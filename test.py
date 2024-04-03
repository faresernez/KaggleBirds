import librosa
import torchaudio
import matplotlib.pyplot as plt 
import numpy as np
import torch.nn as nn
import torch

class cinn(nn.Module):
    # def __init__(self, input_size, hidden_size, output_size):
    def __init__(self):
        super(cinn, self).__init__()
        self.relu = nn.ReLU()
        self.cinn1 = nn.Conv2d(1, 64, 5, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.cinn2 = nn.Conv2d(64, 128, 5, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.cinn3 = nn.Conv2d(128, 256, 5, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        self.cinn4 = nn.Conv2d(256, 256, 3, stride=3, padding=0)
        self.mxp = nn.MaxPool2d(4, stride=2)
        self.lin = nn.Linear(in_features=256, out_features=12)
        self.soft = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.cinn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.cinn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.cinn3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.cinn4(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.mxp(x)
        x = torch.flatten(x, start_dim=1)
        x = self.lin(x)
        x = self.soft(x)
        return 

file = r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\train_audio\abythr1\XC115981.ogg'

data, samplerate = torchaudio.load(file, normalize=True)
print(samplerate)
print(data.shape)
data = np.asarray(data[:128000])
print(data.shape)
y = librosa.feature.mfcc(y = data[0][:64000], sr = samplerate,n_mfcc = 126)
t = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)

# fig, ax = plt.subplots(1,1, figsize = (20,10))
# ax[0].set(title = 'MFCCs of Guitar')
# i = librosa.display.specshow(y, x_axis='time', ax=ax[0])

print(y.shape)

res = cinn()

t = torch.cat((t,t),dim=0)
print(t.shape)
t = res(t)
# print(y)


# file = r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\train_audio\wtbeat1\XC234928.ogg'

# data, samplerate = torchaudio.load(file, normalize=True)
# print(samplerate)
# print(data.shape)
# y = librosa.feature.mfcc(y = np.asarray(data[:64000]), sr = samplerate)

# # fig, ax = plt.subplots(1,1, figsize = (20,10))
# # ax[0].set(title = 'MFCCs of Guitar')
# # i = librosa.display.specshow(y, x_axis='time', ax=ax[0])

# print(y.shape)
# print(y)