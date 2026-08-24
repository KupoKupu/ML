from torch import nn

class CNN(nn.Module):
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
        self.fc2 = nn.Linear(240, 240)
        self.norm4 = nn.BatchNorm1d(240)
        self.norm5 = nn.BatchNorm1d(240)
        self.fc3 = nn.Linear(240, 10)
        self.flat = nn.Flatten()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.ReLU(self.norm1(self.conv1(x))))
        x = self.pool(self.ReLU(self.norm2(self.conv2(x))))
        x = self.ReLU(self.norm3(self.conv3(x)))
        x = self.flat(x)
        x = self.ReLU(self.norm4(self.fc1(x)))
        x = self.ReLU(self.norm5(self.fc2(x)))
        x = self.fc3(x)
        return x

model = CNN()

name = "CNN"