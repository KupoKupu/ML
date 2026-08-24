import torch
from attentionUnet import Network
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
from tqdm import tqdm
import random
import torch.optim as optim
from timm.utils import ModelEmaV3
from typing import List

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cpu'

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

batch_size = 64
# train_data = train_data.data[train_data.targets == 6]
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

class Scheduler(nn.Module):
    def __init__(self, time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, time_steps, requires_grad=False).to(device)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False).to(device)

    def forward(self, t):
        return self.beta[t], self.alpha[t]

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, dim: int):
        super().__init__()
        self.dim = dim
        position = torch.arange(time_steps).unsqueeze(1).float().to(device)
        div = torch.exp(torch.arange(0, dim*dim, 2).float() * -(math.log(10000.0) / (dim*dim))).to(device)
        embeddings = torch.zeros(time_steps, dim*dim, requires_grad=False).to(device)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, t):
        embeds = self.embeddings[t]
        return embeds.view(-1, 1, self.dim, self.dim)

class Epsilon(nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.UNet = Network(1).to(device)
        self.PE = SinusoidalEmbeddings(time_steps, 28).to(device)

    def forward(self, x, t):
        x = x + self.PE(t)
        out = self.UNet(x)
        return out

Epsilon = Epsilon(1000).to(device)

optimizer = torch.optim.Adam(Epsilon.parameters(), lr=2e-5)

def train(epoches, time_steps, dataloader):
    scheduler = Scheduler(time_steps)
    ema = ModelEmaV3(Epsilon, decay=0.9999)

    for i in range(epoches):
        epoch_loss = 0
        print('Epoch', i + 1)
        for (x,_) in tqdm(dataloader):
            x = x.to(device)
            t = torch.randint(0, time_steps, (x.size()[0],)).to(device)
            epsilon = torch.randn_like(x).to(device)
            optimizer.zero_grad()
            bar_alpha = scheduler.alpha[t].view(-1, 1, 1, 1)
            x_t = torch.sqrt(bar_alpha) * x + torch.sqrt(1 - bar_alpha) * epsilon
            ep = Epsilon(x_t, t)
            l = nn.MSELoss()(epsilon, ep)
            l.backward()
            optimizer.step()
            ema.update(Epsilon)
            epoch_loss += l.item()
        average_loss = epoch_loss / len(dataloader)
        print("Epoch average loss", average_loss)

    savings = {
        'weights': Epsilon.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }
    torch.save(savings, 'Ep_savings')


def step(x, t, z):
    ep = Epsilon(x, t)
    scheduler = Scheduler(1000)
    beta = scheduler.beta[t].view(-1, 1, 1, 1)
    bar_alpha = scheduler.alpha[t].view(-1, 1, 1, 1)
    out = 1 / torch.sqrt(1 - beta) * (x - beta / torch.sqrt(1 - bar_alpha)
                                 * ep) + torch.sqrt(beta) * z
    out = out.to(device)
    return out

def generate(x_T, T):
    t = T-1
    x_t = x_T
    while t > 0:
        z = torch.randn_like(x_t).to(device)
        x_t = step(x_t, t, z)
        t = t - 1
    out = step(x_t, t, torch.zeros_like(x_t).to(device))
    return out

# Epsilon.train()
# train(100, 1000, train_loader)

Epsilon.eval()
savings = torch.load('Ep_savings')
Epsilon.load_state_dict(savings['weights'])
ema = ModelEmaV3(Epsilon, decay=0.999)
ema.load_state_dict(savings['ema'])
Epsilon = ema.module.eval()

for i in range(10):
    x_T = torch.randn(1, 1, 28, 28).to(device)
    out = generate(x_T, 1000)[0, 0]
    plt.imshow(out.detach().cpu().numpy())
    plt.show()






