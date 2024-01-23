

import os
import yaml
import torch
import numpy as np
import torch.nn as nn
from skimage import color
import matplotlib.pyplot as plt

from EDA import *
from dataloaders import *
from nets import *
from learning import *
from losses import *
from utils import *

dataset = MakeDataset("./datasets/landscapes/",UsedDataSize='ALL')
device = torch.device("cpu")
network = ECCVGenerator().to(device)
network.load_state_dict(torch.load("./saved/network.pth",map_location=device)) ; network.eval()

pred_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
images, targets = next(iter(pred_loader))

predicted_colors = network.forward(images.float().to(device)).to(device)

true_image = color.lab2rgb(torch.cat((images, targets),1).detach().to(device).numpy().transpose((2,3,1,0))[:,:,:,0]).clip(0,1)
pred_image = color.lab2rgb(torch.cat((images, predicted_colors),1).detach().to(device).numpy().transpose((2,3,1,0))[:,:,:,0]).clip(0,1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
ax1.imshow(true_image) ; ax1.set_title('Original') ; ax1.set_axis_off() 
ax2.imshow(pred_image) ; ax2.set_title('Colorized') ; ax2.set_axis_off()
plt.show()

