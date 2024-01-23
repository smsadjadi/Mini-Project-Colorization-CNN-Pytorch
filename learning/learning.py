

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def train(device,network,optimizer,criterion,dataloader,nb_epochs):
    
    network = network.to(device)
    if os.path.exists("./saved/network.pth"): network.load_state_dict(torch.load("./saved/network.pth",map_location=device)) ; network.train()
    
    if os.path.exists("./saved/report.csv"):
        report = pd.read_csv("./saved/report.csv")
        epochs = np.arange(len(report["loss"]),len(report["loss"])+nb_epochs)
        losses = report["loss"]
    else:
        report = pd.DataFrame(columns=["loss"])
        epochs = np.arange(nb_epochs)
        losses = []
        
    def training_step(device,network,optimizer,criterion,dataloader):

        iteration_losses = []
        train_loop = tqdm(enumerate(dataloader, 1), total=len(dataloader), desc="train", position=0)

        for data in dataloader:
            images, targets = data

            predicted_colors = network.forward(images.float().to(device))
            loss = criterion(predicted_colors, targets.float().to(device))
            iteration_losses.append(loss.item())
            optimizer.zero_grad() 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),10) # clip gradients to 10
            optimizer.step()
            avgloss = sum(iteration_losses) / len(iteration_losses) # average loss of training iteration

            train_loop.update(1)
            train_loop.set_postfix(loss="{:.4f}".format(avgloss),refresh=True)

        return avgloss

    for epoch in epochs:
        print('epoch',epoch+1,'----------------------------------------')
        lastloss = training_step(device,network,optimizer,criterion,dataloader)
        losses = np.append(losses,lastloss)
        report = pd.DataFrame({"loss":losses}) ; report.to_csv("./saved/report.csv")
        torch.save(network.state_dict(),"./saved/network.pth")

