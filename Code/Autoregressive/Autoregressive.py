import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x >= 0.5).float())
])

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# train_data = train_data.data[train_data.targets == 6]

train_dataloader = DataLoader(train_data, batch_size=300, shuffle=True)


class NADE(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.W = nn.Linear(28 * 28, self.h_dim)
        self.sigmoid = nn.Sigmoid()
        self.linears = nn.ModuleList([nn.Linear(h_dim, 1) for _ in range(28 * 28)])

    def zero_pad(self, x, n):
        batch_size = x.size()[0]
        out = torch.concat((x[:,:n], torch.zeros(batch_size, 28 * 28 - n).to(device)), dim=1)
        return out

    def forward(self, x):
        batch_size = x.size()[0]
        out = torch.zeros(28 * 28, batch_size).to(device)
        for i in range(28 * 28):
            h = self.sigmoid(self.W(self.zero_pad(x, i)))
            out[i] = self.sigmoid(self.linears[i](h)).view(batch_size)
        out = torch.transpose(out, 0, 1)
        neg_log_likelihood = nn.BCELoss(reduction='sum')(out, x)
        return neg_log_likelihood

    def sample(self):
        x = torch.zeros(1, 28 * 28).to(device)
        theta = torch.zeros(1, 28 * 28).to(device)
        for i in range(28 * 28):
            h = self.sigmoid(self.W(x))
            next_theta = self.sigmoid(self.linears[i](h))
            next_one = torch.bernoulli(next_theta)
            theta[0][i] = next_theta[0]
            x[0][i] = next_one
        return x, theta


AR = NADE(1024).to(device)
optimizer = torch.optim.Adam(AR.parameters(), lr=0.001)

EPOCHS = 10

def train(dataloader):
    for j in range(EPOCHS):
        running_loss = 0.
        i = 0
        print('Epoch', j + 1)
        for (X, _) in tqdm(dataloader):
            X = X.to(device)
            # X = (X / 255 > 0.5).float()
            X = X.view(-1, 28 * 28)
            optimizer.zero_grad()
            loss = AR(X)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
                last_loss = running_loss / (50 * X.size()[0])
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.
            i += 1

def load():
    file_path = Path('AR.pth')
    if file_path.is_file():
        cp = torch.load('AR.pth')
        optimizer.load_state_dict(cp['optimizerAR'])
        AR.load_state_dict(cp['modelAR'])

def save():
    cp = {
        'modelAR': AR.state_dict(),
        'optimizerAR': optimizer.state_dict(),
    }
    torch.save(cp, 'AR.pth')

AR.train()
load()
train(train_dataloader)
save()

AR.eval()
load()

def sample(n):
    out = torch.zeros(n ** 2, 28, 28).to(device)
    for i in range(n ** 2):
        out[i] = AR.sample()[1].view(28, 28)
    fig = plt.figure()
    columns = n
    rows = n
    for i in range(1, columns * rows + 1):
        img = out[i - 1].cpu().detach().numpy()
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

sample(5)
