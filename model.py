import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool2d(2),   # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), # 128@18*18
            nn.MaxPool2d(2), # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),   # 256@6*6
        )
        self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out


class Siamese_mfcc(nn.Module):

    # mfcc size: (batch,1,13,2500) 

    def __init__(self):
        super(Siamese_mfcc, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10,  padding = True, padding_mode='zeros'),  # 64@13*2500
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(2),  
            nn.Conv2d(64, 64, 7, padding = True,padding_mode='zeros'),
            nn.ReLU(),    # 64@13*2500
            #nn.MaxPool2d(2),   
            nn.Conv2d(32, 64, 4, padding = True,padding_mode='zeros'),
            nn.ReLU(), # 32@13*2500
            #nn.MaxPool2d(2), 
            nn.Conv2d(16, 64, 4, padding = True, padding_mode='zeros'),
            nn.ReLU(),   # 16@13*2500
        )
        self.liner = nn.Sequential(nn.Linear(520000, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)


    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x


    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out

# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))
