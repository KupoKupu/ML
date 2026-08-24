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

train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

class ProbEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 4)
        self.conv3 = nn.Conv2d(256, 512, 2)
        self.fc1 = nn.Linear(512 * 4 * 4, 240)
        self.norm1 = nn.LayerNorm([128, 26, 26])
        self.norm2 = nn.LayerNorm([256, 10, 10])
        self.norm3 = nn.LayerNorm([512, 4, 4])
        self.fc2 = nn.Linear(480, 240)
        self.norm4 = nn.LayerNorm(240)
        self.fc_var1 = nn.Linear(240, 512)
        self.fc_var2 = nn.Linear(512, 28 * 28)
        self.fc_mean1 = nn.Linear(240, 512)
        self.fc_mean2 = nn.Linear(512, 28 * 28)
        self.flat = nn.Flatten()
        self.ReLU = nn.LeakyReLU()
        self.fc1_y = nn.Linear(1, 120)
        self.fc2_y = nn.Linear(120, 240)


    def forward(self, x, y):
        x = self.pool(self.ReLU(self.norm1(self.conv1(x))))
        x = self.pool(self.ReLU(self.norm2(self.conv2(x))))
        x = self.ReLU(self.norm3(self.conv3(x)))
        x = self.flat(x)
        x = self.ReLU(self.norm4(self.fc1(x)))
        y = self.ReLU(self.fc1_y(y))
        y = self.ReLU(self.fc2_y(y))
        out = torch.concat((x, y), dim = 1)
        out = self.ReLU(self.fc2(out))
        log_var =self.ReLU(self.fc_var1(out))
        log_var = self.fc_var2(log_var)
        mean = self.ReLU(self.fc_mean1(out))
        mean = self.fc_mean2(mean)
        return log_var, mean

class ProbDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 4)
        self.conv3 = nn.Conv2d(256, 512, 2)
        self.fc1 = nn.Linear(512 * 4 * 4, 240)
        self.norm1 = nn.LayerNorm([128, 26, 26])
        self.norm2 = nn.LayerNorm([256, 10, 10])
        self.norm3 = nn.LayerNorm([512, 4, 4])
        self.fc2 = nn.Linear(480, 240)
        self.norm4 = nn.LayerNorm(240)
        self.fc3 = nn.Linear(240, 512)
        self.fc4 = nn.Linear(512, 28 * 28)
        self.flat = nn.Flatten()
        self.ReLU = nn.LeakyReLU()
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
        out = self.ReLU(self.fc2(out))
        out = self.ReLU(self.fc3(out))
        out = self.sigmoid(self.fc4(out)).view(-1, 1, 28, 28)
        return out

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ProbEncoder = ProbEncoder()
        self.ProbDecoder = ProbDecoder()

    def reparametrize(self, mean, log_var):
        e = torch.normal(mean=torch.zeros(28 * 28), std=torch.ones(28 * 28)).to(device)
        z = mean + torch.exp(0.5 * log_var) * e
        return z

    def forward(self, x, y):
        mean, log_var = self.ProbEncoder(x, y)
        z = self.reparametrize(mean, log_var)
        z = z.view(-1, 1, 28, 28)
        out = self.ProbDecoder(y, z)
        return out

    def neg_KL(self, x, y):
        mean, log_var = self.ProbEncoder(x, y)
        s = 1 + log_var - torch.pow(mean, 2) - torch.exp(log_var)
        s = torch.sum(s) / 2
        return s

    def neg_recon_prob(self, x, y):
        out = self.forward(x, y)
        l = nn.BCELoss(reduction='sum')(out, x)
        return l

    def neg_ELBO(self, x, y):
        l = - self.neg_KL(x, y) / 50 + self.neg_recon_prob(x, y)
        return l

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters())

EPOCHS = 30

def train(dataloader):
    for j in range(EPOCHS):
        running_loss = 0.
        i = 0
        print('Epoch', j + 1)
        for (X, y) in dataloader:
            X = X.to(device)
            batch_size = X.shape[0]
            y = y.to(device)
            y = y.to(torch.float)
            y = y.view(batch_size, 1)
            optimizer.zero_grad()
            loss = model.neg_ELBO(X, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / (100 * X.size()[0]) # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.
            i += 1

model.train()
model.load_state_dict(torch.load('VAE_c.pth', weights_only=True))
train(train_dataloader)
torch.save(model.state_dict(), 'VAE_c.pth')

model.eval()
model.load_state_dict(torch.load('VAE_c.pth', weights_only=True))

noise = torch.normal(0, 1, size=(100, 1, 28, 28)).to(device)
y = torch.zeros(10, 1)

for i in range(9):
    j = (i + 1) * torch.ones(10, 1)
    y = torch.cat((y, j), dim=0)

y = y.to(device)
out = model.ProbDecoder(y, noise).view(-1, 28, 28)
fig = plt.figure()
columns = 10
rows = 10

for i in range(1, columns * rows + 1):
    img = out[i - 1].cpu().detach().numpy()
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis('off')

plt.show()

