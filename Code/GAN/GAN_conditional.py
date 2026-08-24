import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.utils import spectral_norm

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

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

class G_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 4)
        self.conv3 = nn.Conv2d(256, 512, 2)
        self.fc1 = nn.Linear(512 * 4 * 4, 240)
        self.norm1 = nn.BatchNorm2d(128)
        self.norm2 = nn.BatchNorm2d(256)
        self.norm3 = nn.BatchNorm2d(512)
        self.fc2 = nn.Linear(480, 240)
        self.norm4 = nn.BatchNorm1d(240)
        self.norm5 = nn.BatchNorm1d(240)
        self.fc3 = nn.Linear(240, 28 * 28)
        self.flat = nn.Flatten()
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1_y = nn.Linear(1, 120)
        self.fc2_y = nn.Linear(120, 240)


    def forward(self, y, z):
        z = self.pool(self.ReLU(self.norm1(self.conv1(z))))
        z = self.pool(self.ReLU(self.norm2(self.conv2(z))))
        z = self.ReLU(self.norm3(self.conv3(z)))
        z = self.flat(z)
        z = self.ReLU(self.norm4(self.fc1(z)))
        y = self.ReLU(self.fc1_y(y))
        y = self.ReLU(self.fc2_y(y))
        out = torch.concat((z, y), dim = 1)
        out = self.ReLU(self.norm5(self.fc2(out)))
        out = self.sigmoid(self.fc3(out)).view(-1, 1, 28, 28)
        return out

G = G_Net().to(device)

class D_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(1, 128, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = spectral_norm(nn.Conv2d(128, 256, 4))
        self.conv3 = spectral_norm(nn.Conv2d(256, 512, 2))
        self.fc1 = spectral_norm(nn.Linear(512 * 4 * 4, 240))
        self.fc2 = spectral_norm(nn.Linear(480, 240))
        self.fc3 = spectral_norm(nn.Linear(240, 1))
        self.flat = nn.Flatten()
        self.ReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc_y = nn.Linear(1, 120)
        self.fc1_y = nn.Linear(120, 240)

    def forward(self, x, y):
        x = self.pool(self.ReLU(self.conv1(x)))
        x = self.pool(self.ReLU(self.conv2(x)))
        x = self.ReLU(self.conv3(x))
        x = self.flat(x)
        x = self.ReLU(self.fc1(x))
        y = self.ReLU(self.fc_y(y))
        y = self.ReLU(self.fc1_y(y))
        x = torch.concat((x, y), dim = 1)
        x = self.ReLU(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

D = D_Net().to(device)

D_opti = torch.optim.SGD(D.parameters())
G_opti = torch.optim.Adam(G.parameters())

loss = nn.BCELoss(reduction='sum')

def D_step(x, y, z):
    batch_size = x.shape[0]
    ones = torch.ones(batch_size, 1).to(device)
    D_opti.zero_grad()
    loss_D = loss(D(x, y), ones) + loss(1 - D(G(y, z), y), ones)
    loss_D.backward()
    D_opti.step()
    return None


def G_step(y, z):
    batch_size = z.shape[0]
    ones = torch.ones(batch_size, 1).to(device)
    G_opti.zero_grad()
    loss_G = loss(D(G(y, z), y), ones)
    loss_G.backward()
    G_opti.step()
    return None

def train(num_epochs, G_update_times, D_update_times):
    D.train()
    G.train()
    for epoch in range(num_epochs):
        print("Ep:", epoch + 1)
        Real = 0
        Fake = 0
        for (X, y) in tqdm(dataloader):
            X = X.to(device)
            batch_size = X.shape[0]
            y = y.to(device)
            y = y.to(torch.float)
            y = y.view(batch_size, 1)
            for _ in range(D_update_times):
                Z = torch.normal(0, 1, size=(batch_size, 1, 28, 28)).to(device)
                D_step(X, y, Z)
            for _ in range(G_update_times):
                Z = torch.normal(0, 1, size=(batch_size, 1, 28, 28)).to(device)
                G_step(y, Z)
            Real += torch.sum(D(X, y).detach())
            Z = torch.normal(0, 1, size=(batch_size, 1, 28, 28)).to(device)
            Fake += torch.sum(D(G(y, Z), y).detach())
        print("Discriminator on real:", Real / len(train_data))
        print("Discriminator on fake:", Fake / len(train_data))

# D.load_state_dict(torch.load('D_weights_c.pth'))
# G.load_state_dict(torch.load('G_weights_c.pth'))
#
# train(100, 1, 1)
#
# torch.save(D.state_dict(), 'D_weights_c.pth')
# torch.save(G.state_dict(), 'G_weights_c.pth')

G.eval()
G.load_state_dict(torch.load('G_weights_c.pth', weights_only=True))

noise = torch.normal(0, 1, size=(100, 1, 28, 28)).to(device)
y = torch.zeros(10, 1)

for i in range(9):
    j = (i + 1) * torch.ones(10, 1)
    y = torch.cat((y, j), dim = 0)

y = y.to(device)
out = G(y, noise).view(-1, 28, 28)
fig = plt.figure()
columns = 10
rows = 10

for i in range(1, columns * rows + 1):
    img = out[i-1].cpu().detach().numpy()
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis('off')
    
plt.show()


