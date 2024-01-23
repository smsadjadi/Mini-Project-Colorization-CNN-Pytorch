

import os
import yaml
import torch
import torch.nn as nn

from EDA import *
from dataloaders import *
from nets import *
from learning import *
from losses import *
from utils import *

with open('config.yaml') as f: config = yaml.load(f, Loader=yaml.FullLoader)
initial_lr = config['initial_lr']
batch_size = config['batch_size']
nb_epochs = config['nb_epochs']

dataset = MakeDataset("./datasets/landscapes/",UsedDataSize='ALL')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = ECCVGenerator()
optimizer = torch.optim.Adam(network.parameters(), lr = initial_lr , weight_decay=1e-3)
criterion = nn.MSELoss()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

train(device,network,optimizer,criterion,dataloader,nb_epochs)
PlotLoss("./saved/report.csv")

