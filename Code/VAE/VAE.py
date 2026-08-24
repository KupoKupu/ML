import torch
from torch import nn
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

train_data = train_data.data[train_data.targets == 6]

Batch_size = 8

train_dataloader = DataLoader(train_data, Batch_size)

class ProbEncoder(nn.Module):
    def __init__(self, dim, en_dim):
        super().__init__()
        self.dim = dim
        self.en_dim = en_dim
        self.fc1 = nn.Linear(dim, en_dim)
        self.fc2 = nn.Linear(en_dim, en_dim)
        self.fc3 = nn.Linear(en_dim, en_dim)

    def forward(self, x):
        h = self.fc1(x)
        h = nn.Tanh()(h)
        log_var = self.fc2(h)
        mean = self.fc3(h)
        return log_var, mean

class ProbDecoder(nn.Module):
    def __init__(self, dim, en_dim):
        super().__init__()
        self.dim = dim
        self.en_dim = en_dim
        self.fc1 = nn.Linear(en_dim, en_dim)
        self.fc2 = nn.Linear(en_dim, dim)

    def forward(self, z):
        h = self.fc1(z)
        h = nn.Tanh()(h)
        y = self.fc2(h)
        y = nn.Sigmoid()(y)
        return y

class VAE(nn.Module):
    def __init__(self, dim, en_dim):
        super().__init__()
        self.dim = dim
        self.en_dim = en_dim
        self.ProbEncoder = ProbEncoder(dim, en_dim)
        self.ProbDecoder = ProbDecoder(dim, en_dim)

    def reparametrize(self, mean, log_var):
        en_dim = self.en_dim
        z = mean + torch.exp(0.5 * log_var) * torch.normal(mean=torch.zeros(en_dim), std=torch.ones(en_dim)).to(device)
        return z

    def forward(self, x):
        mean, log_var = self.ProbEncoder(x)
        z = self.reparametrize(mean, log_var)
        y = self.ProbDecoder(z)
        return y

    def KL(self, x):
        mean, log_var = self.ProbEncoder(x)
        s = 1 + log_var - torch.pow(mean, 2) - torch.exp(log_var)
        s = torch.sum(s) / 2
        return s

    def recon_error(self, x):
        y = self.forward(x)
        l = nn.functional.binary_cross_entropy(y, x, reduction='sum')
        return l

    def Loss(self, x):
        l = - self.KL(x) + self.recon_error(x)
        return l

model = VAE(784, 1024).to(device)
optimizer = torch.optim.Adam(model.parameters())

EPOCHS = 300

def train(dataloader):
    for j in range(EPOCHS):
        running_loss = 0.
        i = 0
        print('Epoch', j + 1)
        for x in dataloader:
            x = x / 255.0
            x = x.view(-1, 784).to(device)
            optimizer.zero_grad()
            loss = model.Loss(x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / (100 * x.size()[0]) # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.
            i += 1

# model.train()
# model.load_state_dict(torch.load('VAE.pth', weights_only=True))
# train(train_dataloader)
# torch.save(model.state_dict(), 'VAE.pth')

model.eval()
model.load_state_dict(torch.load('VAE.pth', weights_only=True))

with torch.no_grad():
    noise = torch.randn(8, 1024).to(device)
    generated_images = model.ProbDecoder(noise).view(8, 28, 28)
    plt.imshow(generated_images[0].cpu().numpy())
    plt.show()

