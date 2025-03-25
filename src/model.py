import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.init as init

class NeuralNetwork(nn.Module):
    def __init__(self, weight_init='random'):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(3, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 16)
        self.layer5 = nn.Linear(16, 4)
        self.activation = nn.ReLU()

        if weight_init =='xavier':
            nn.init.xavier_uniform_(self.layer1.weight)
            nn.init.xavier_uniform_(self.layer2.weight)
            nn.init.xavier_uniform_(self.layer3.weight)
            nn.init.xavier_uniform_(self.layer4.weight)
            nn.init.xavier_uniform_(self.layer5.weight)
        else:
            # Random initialization (default)
            nn.init.normal_(self.layer1.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.layer2.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.layer3.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.layer4.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.layer5.weight, mean=0.0, std=0.1) 

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.activation(self.layer4(x))
        x = self.layer5(x)
        return x