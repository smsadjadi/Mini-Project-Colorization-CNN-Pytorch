

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

class BaseColor(nn.Module):

  def __init__(self):
      super(BaseColor, self).__init__()
      self.l_cent = 50.
      self.l_norm = 100.
      self.ab_norm = 110.
      
  def normalize_l(self, in_l): return (in_l-self.l_cent)/self.l_norm
  def unnormalize_l(self, in_l): return in_l*self.l_norm + self.l_cent 
  def normalize_ab(self, in_ab): return in_ab/self.ab_norm
  def unnormalize_ab(self, in_ab): return in_ab*self.ab_norm

def PlotLoss(reportpath = "./saved/report.csv"):
    report = pd.read_csv(reportpath)
    losses = report['loss']
    plt.figure(figsize=(4,3))
    plt.plot(np.arange(len(losses))+1 ,losses, label="train loss")
    plt.title('Loss per Epoch') ; plt.xlabel('epochs') ; plt.ylabel('Avg Loss')
    plt.legend() ; plt.show()

