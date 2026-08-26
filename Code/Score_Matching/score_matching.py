import torch
from attentionUnet import Network
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import math


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

batch_size = 80
# train_data = train_data.data[train_data.targets == 6]
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

class PositionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        pe_matrix= torch.zeros(28, 28)
        for pos in range(28):
            for i in range(28):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / 28)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / 28)))
        pe_matrix = pe_matrix.unsqueeze(0)
        self.positional_encoding = pe_matrix.to(device).requires_grad_(False)

    def forward(self, x):
        x = x + self.positional_encoding
        return x

class Score(nn.Module):
    def __init__(self):
        super().__init__()
        self.Unet = Network(1)
        self.PE = PositionalEncoder()

    def forward(self, x, z, sigma):
        out = self.Unet(self.PE(x + sigma * z))
        return out

    def sample(self, x):
        out = self.Unet(self.PE(x))
        return out

class multi_score(nn.Module):
    def __init__(self, n_levels, ratio, sigma):
        super().__init__()
        self.score = Score()
        self.sigma = sigma
        self.n_levels = n_levels
        self.ratio = ratio

    def forward(self, x):
        L = 0
        for i in range(self.n_levels):
            z = torch.randn_like(x).to(device)
            sigma = self.sigma * (self.ratio ** i)
            loss = nn.MSELoss(reduction='mean')(self.score(x, z, sigma), - z / sigma)
            L += (sigma ** 2) * loss
        return L

model = multi_score(10, 0.9, 0.5).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def load():
    file_path = Path('score_savings')
    if file_path.is_file():
        cp = torch.load('score_savings')
        optimizer.load_state_dict(cp['optimizer'])
        model.load_state_dict(cp['model'])

def save():
    cp = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(cp, 'score_savings')

def train(epoches, dataloader):
    for i in range(epoches):
        epoch_loss = 0
        print('Epoch', i + 1)
        for (x, _) in tqdm(dataloader):
            x = x.to(device)
            # x = (x / 255).float().view(-1, 1, 28, 28)
            optimizer.zero_grad()
            l = model(x)
            l.backward()
            optimizer.step()
            epoch_loss += l.detach().item()
        average_loss = epoch_loss / len(dataloader)
        print("Epoch average loss", average_loss)


model.train()
load()
train(100, train_loader)
save()

model.eval()

def generate(x0, n_iters, ep):
    x = x0
    for _ in range(n_iters):
        z = torch.randn_like(x).to(device)
        x = x + ep * model.score.sample(x).detach() + ((ep * 2) ** 0.5) * z
    return x

def sample(n, n_iters, ep):
    x0 = torch.randn(n ** 2, 1, 28, 28).to(device)
    out = generate(x0, n_iters, ep).view(-1, 28, 28)
    fig = plt.figure()
    columns = n
    rows = n
    for i in range(1, columns * rows + 1):
        img = out[i - 1].cpu().detach().numpy()
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

sample(5, 500, 1e-2)