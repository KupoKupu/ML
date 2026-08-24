import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torch import distributions
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
from tqdm import tqdm
import random
import torch.optim as optim
from timm.utils import ModelEmaV3
from typing import List
from batch_norm import MyBatchNorm2d

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
    transform=transforms.ToTensor())

train_data = train_data.data[train_data.targets == 6]
train_data = train_data / 255

train_loader = DataLoader(train_data, batch_size=50, shuffle=True)

class NetS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.bnorm1 = nn.BatchNorm2d(128, affine=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 4)
        self.bnorm2 = nn.BatchNorm2d(256, affine=False)
        self.conv3 = nn.Conv2d(256, 256, 4)
        self.bnorm3 = nn.BatchNorm2d(256, affine=False)
        self.flat = nn.Flatten()
        self.fc2 = nn.Linear(256 * 2 * 2, 28 * 28)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.pool(self.tanh(x))
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.pool(self.tanh(x))
        x = self.conv3(x)
        x = self.bnorm3(x)
        x = self.tanh(x)
        x = self.flat(x)
        x = 1.5 * self.tanh(self.fc2(x))
        return x


class NetT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.bnorm1 = nn.BatchNorm2d(128, affine=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 4)
        self.bnorm2 = nn.BatchNorm2d(256, affine=False)
        self.conv3 = nn.Conv2d(256, 256, 4)
        self.bnorm3 = nn.BatchNorm2d(256, affine=False)
        self.flat = nn.Flatten()
        self.fc2 = nn.Linear(256 * 2 * 2, 28 * 28)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.pool(self.tanh(x))
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.pool(self.tanh(x))
        x = self.conv3(x)
        x = self.bnorm3(x)
        x = self.tanh(x)
        x = self.flat(x)
        x = 1.5 * self.tanh(self.fc2(x))
        return x

class CouplingLayer(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask
        self.netS = NetS()
        self.netT = NetT()

    def forward(self, x):
        x = x.view(-1, 28**2)
        out = self.mask * x + (1 - self.mask) * (x * torch.exp(self.netS(self.mask * x)) + self.netT(self.mask * x))
        log_det = ((1 - self.mask) * self.netS(self.mask * x)).sum(dim=1)
        return out, log_det

    def inverse(self, x):
        x = x.view(-1, 28**2)
        out = self.mask * x + (1 - self.mask) * (x - self.netT(self.mask * x)) * torch.exp(-self.netS(self.mask * x))
        return out

class Layer(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask
        self.clA = CouplingLayer(mask)
        self.clB = CouplingLayer(1 - mask)
        # self.bnormA = MyBatchNorm2d(1)
        # self.bnormB = MyBatchNorm2d(1)

    def forward(self, x):
        out, log_det_A = self.clA(x)
        # out = out.view(-1, 1, 28, 28)
        # out, log_det_normA = self.bnormA(out)
        out, log_det_B = self.clB(out)
        # out = out.view(-1, 1, 28, 28)
        # out, log_det_normB = self.bnormB(out)
        return out, log_det_A + log_det_B

    def inverse(self, x):
        # x = x.view(-1, 1, 28, 28)
        # out, _ = self.bnormB(x)
        out = self.clB.inverse(x)
        # out = out.view(-1, 1, 28, 28)
        # out, _ = self.bnormA(out)
        out = self.clA.inverse(out)
        return out


class FlowNVP(nn.Module):
    def __init__(self, masks):
        super().__init__()
        self.masks = masks
        self.layers = nn.ModuleList([Layer(mask) for mask in masks])
        self.std_normal = distributions.MultivariateNormal(torch.zeros(28**2).to(device), torch.eye(28**2).to(device))

    def forward(self, x):
        log_det_out = torch.zeros(x.size()[0]).to(device)
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_out += log_det
        neg_log_p = - log_det_out - self.std_normal.log_prob(x.view(-1, 28 * 28))
        # print("-det:", -log_det_out.sum())
        # print("-log p(x):", -self.std_normal.log_prob(x).sum())
        return x, neg_log_p

    def inverse(self, x):
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x


checkerboard_mask = torch.zeros(28 * 28).to(device)
checkerboard_mask[::2] = 1

masks = [checkerboard_mask] * 3
model = FlowNVP(masks).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

EPOCHS = 50

def train(dataloader):
    for j in range(EPOCHS):
        running_loss = 0.
        for x in tqdm(dataloader):
            x = torch.squeeze(x).to(device)
            optimizer.zero_grad()
            loss = model(x)[1].mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        last_loss = running_loss / len(dataloader)
        print('Epoch {} Training Loss: {}'.format(j + 1, last_loss))



model.train()

checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# train(train_loader)
#
# checkpoint = {
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
# }
# torch.save(checkpoint, 'checkpoint.pth')

model.eval()

with torch.no_grad():
    noise = torch.Tensor(torch.normal(torch.zeros(1, 28 * 28),
                                  torch.ones(1, 28 * 28))).to(device)
    x = torch.squeeze(train_data[0]).to(device)
    original_image = x.view(28, 28)
    y = model(x)[0]
    generated_images = model.inverse(noise).view(1, 28, 28)
    generated_images = torch.clip(generated_images, min=0, max=1)
    plt.imshow(generated_images[0].cpu().numpy())
    plt.show()
    # plt.imshow(original_image.cpu().numpy())
    # plt.show()