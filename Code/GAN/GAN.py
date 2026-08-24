import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

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

train_data = train_data.data[train_data.targets == 7]

dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

G = nn.Sequential(
    nn.Linear(1024, 512), nn.Tanh(),
    nn.Linear(512, 512), nn.Tanh(),
    nn.Linear(512, 512), nn.Tanh(),
    nn.Linear(512, 784), nn.Sigmoid()).to(device)

D = nn.Sequential(
    nn.Linear(784, 256), nn.Tanh(),
    nn.Linear(256, 64), nn.Tanh(),
    nn.Linear(64, 1), nn.Sigmoid()).to(device)

D_opti = torch.optim.Adam(D.parameters())
G_opti = torch.optim.Adam(G.parameters())

loss = nn.BCELoss(reduction='sum')

def D_step(X, Z):
    batch_size = X.shape[0]
    ones = torch.ones(batch_size, 1).to(device)
    D_opti.zero_grad()
    loss_D = loss(D(X), ones) + loss(1 - D(G(Z)), ones)
    loss_D.backward()
    D_opti.step()
    return None


def G_step(Z):
    batch_size = Z.shape[0]
    ones = torch.ones(batch_size, 1).to(device)
    G_opti.zero_grad()
    loss_G = loss(D(G(Z)), ones)
    loss_G.backward()
    G_opti.step()
    return None

def train(num_epochs, G_update_times, D_update_times):
    D.train()
    G.train()
    for epoch in range(num_epochs):
        print("Ep:", epoch+1)
        Real = 0
        Fake = 0
        for X in dataloader:
            X = X / 255.0
            X = X.to(device)
            X = X.view(-1, 784)
            batch_size = X.shape[0]
            for _ in range(D_update_times):
                Z = torch.normal(0, 1, size=(batch_size, 1024)).to(device)
                D_step(X, Z)
            for _ in range(G_update_times):
                Z = torch.normal(0, 1, size=(batch_size, 1024)).to(device)
                G_step(Z)
            Real += torch.sum(D(X).detach())
            Z = torch.normal(0, 1, size=(batch_size, 1024)).to(device)
            Fake += torch.sum(D(G(Z)).detach())
        print("Discriminator on real:", Real / len(train_data))
        print("Discriminator on fake:", Fake / len(train_data))


train(100, 1, 1)
torch.save(D.state_dict(), 'D_weights.pth')
torch.save(G.state_dict(), 'G_weights.pth')

D.eval()
D.load_state_dict(torch.load('D_weights.pth', weights_only=True))
G.eval()
G.load_state_dict(torch.load('G_weights.pth', weights_only=True))

noise = torch.normal(0, 1, size=(25, 1024)).to(device)
out = G(noise).view(-1, 28, 28)
fig = plt.figure()
columns = 5
rows = 5
for i in range(1, columns * rows + 1):
    img = out[i-1].cpu().detach().numpy()
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

