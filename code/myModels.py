import torch.nn as nn
import torch
from ULite import ULite
from torchsummary import summary



class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 64, 5, stride=3, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(288, 128)
        )
         
        self.decoder = nn.Sequential(
            nn.Linear(128, 288),
            nn.ReLU(),
            nn.Unflatten(1, (32, 3, 3)),  # Assuming the output of the last conv layer was (32, 3, 3)
            nn.ConvTranspose2d(32, 64, 3, stride=2, padding=0, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, 5, stride=3, padding=0, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=0, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid() 
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class Classifier(nn.Module):

    def __init__(self,autoEncoder,inputSize,nClasses):
        super(Classifier,self).__init__()
        self.nClasses = nClasses
        self.inputSize = inputSize
        self.autoEncoder = autoEncoder

        self.network = nn.Sequential(
            autoEncoder.encoder,
            nn.Linear(self.inputSize,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,self.nClasses),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

class ClassifierForULite(nn.Module):

    def __init__(self,autoEncoder,nClasses):

        super(ClassifierForULite,self).__init__()
        self.nClasses = nClasses
        
        self.autoEncoder = autoEncoder

        self.totunecnn1 = nn.Conv2d(512, 16, 3, stride=1, padding=0, bias=False)
        self.totunecnn2 = nn.Conv2d(16, 8, 3, stride=1, padding=0, bias=False)
        # self.totunecnn1 = nn.Conv2d(512, 64, 3, stride=1, padding=0, bias=False)
        # self.totunecnn2 = nn.Conv2d(64, 32, 3, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3)
        # self.lin1 = nn.Linear(1024,512)
        # self.lin2 = nn.Linear(512,256)
        # self.lin3 = nn.Linear(256,128)
        # self.lin4 = nn.Linear(128,64)
        # self.lin5 = nn.Linear(64,32)
        self.totunelin6 = nn.Linear(8,self.nClasses)
        # self.soft = nn.Softmax(dim=1)

        # self.network = nn.Sequential(
        #     # autoEncoder.encoder,
        #     nn.Conv2d(512, 256, 5, stride=1, padding=0, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 128, 3, stride=1, padding=0, bias=False),

        #     nn.Linear(self.inputSize,64),
        #     nn.ReLU(),
        #     nn.Linear(64,32),
        #     nn.ReLU(),
        #     nn.Linear(32,self.nClasses),
        #     nn.Softmax(dim=1)

    def forward(self, x):
        x = self.autoEncoder.conv_in(x)
        x, _ = self.autoEncoder.e1(x)
        x, _ = self.autoEncoder.e2(x)
        x, _ = self.autoEncoder.e3(x)
        x, _ = self.autoEncoder.e4(x)
        x, _ = self.autoEncoder.e5(x)
        x = self.totunecnn1(x)
        x = self.relu(x)
        x = self.totunecnn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        # x = self.lin1(x)
        # x = self.relu(x)
        # x = self.lin2(x)
        # x = self.relu(x)
        # x = self.lin3(x)
        # x = self.relu(x)
        # x = self.lin4(x)
        # x = self.relu(x)
        # x = self.lin5(x)
        # x = self.relu(x)
        x = self.totunelin6(x)
        # x = self.relu(x)
        # x = self.soft(x)
        return x


# class AE(nn.Module):
#     def __init__(self):
#         super(AE,self).__init__()
#         self.relu = nn.ReLU()
#         self.cnn1 = nn.Conv2d(1, 64, 5, stride=2, padding=0, bias=False)
#         self.cnn2 = nn.Conv2d(64, 128, 5, stride=2, padding=0, bias=False)
#         self.cnn3 = nn.Conv2d(128, 128, 3, stride=2, padding=0, bias=False)
#         self.cnn4 = nn.Conv2d(128, 64, 5, stride=3, padding=0, bias=False)
#         self.cnn5 = nn.Conv2d(64, 32, 3, stride=2, padding=0, bias=False)
#         self.flatten = nn.Flatten(start_dim=1)
#         self.lin1 = nn.Linear(288, 128)
#         self.lin2 = nn.Linear(128, 288)
#         self.unflatten = nn.Unflatten(1, (32, 3, 3))
#         self.cnntrans1 = nn.ConvTranspose2d(32, 64, 3, stride=2, padding=0, output_padding=1)
#         self.cnntrans2 = nn.ConvTranspose2d(64, 128, 5, stride=3, padding=0, output_padding=1)
#         self.cnntrans3 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=0, output_padding=1)
#         self.cnntrans4 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)
#         self.cnntrans5 = nn.ConvTranspose2d(64, 1, 5, stride=2, padding=2, output_padding=1)
#         self.sig = nn.Sigmoid()

#     def forward(self,x):
#         print(x.shape)
#         x = self.cnn1(x)
#         print(x.shape)
#         x = self.relu(x)
#         x = self.cnn2(x)
#         print(x.shape)
#         x = self.relu(x)
#         x = self.cnn3(x)
#         print(x.shape)
#         x = self.relu(x)
#         x = self.cnn4(x)
#         print(x.shape)
#         x = self.relu(x)
#         x = self.cnn5(x)
#         print(x.shape)
#         x = self.relu(x)
#         x = self.flatten(x)
#         print('after flatten')
#         print(x.shape)
#         x = self.lin1(x)
#         print(x.shape)
#         x = self.lin2(x)
#         print(x.shape)
#         x = self.relu(x)
#         x = self.unflatten(x)
#         print(x.shape)
#         print('after unflatten')
#         x = self.cnntrans1(x)
#         print(x.shape)
#         x = self.relu(x)
#         x = self.cnntrans2(x)
#         print(x.shape)
#         x = self.relu(x)
#         x = self.cnntrans3(x)
#         print(x.shape)
#         x = self.relu(x)
#         x = self.cnntrans4(x)
#         print(x.shape)
#         x = self.relu(x)
#         x = self.cnntrans5(x)
#         print(x.shape)
#         x = self.sig(x)
#         return x

### TESTS ###

# pretrainedModel = ULite().to('cuda:0')
# model = ClassifierForULite(pretrainedModel,3).to('cuda:0')
# summary(model, (1,224,224))
    
# model = AE()
# t= torch.ones((224,224)).unsqueeze(0).unsqueeze(0)
# a= torch.zeros((224,224)).unsqueeze(0).unsqueeze(0)
# x = torch.cat((t,a),dim=0)
# print(x.shape)
# res = model.forward(x)
# print(res.shape)