from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, x_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.x_dim = x_dim
        self.conv0 = nn.Conv2d(in_dim, out_dim, 1)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3)
        self.conv3 = nn.Conv2d(out_dim, out_dim, 3)
        self.norm1 = nn.BatchNorm2d(out_dim)
        self.norm2 = nn.BatchNorm2d(out_dim)
        self.norm3 = nn.BatchNorm2d(out_dim)
        self.fc1 = nn.Linear(x_dim ** 2, (x_dim - 2) ** 2)
        self.fc2 = nn.Linear((x_dim - 2) ** 2, (x_dim - 4) ** 2)
        self.fc3 = nn.Linear((x_dim - 4) ** 2, (x_dim - 6) ** 2)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        y = self.ReLU(self.norm1(self.conv1(x)))
        x = self.conv0(x).view(-1, self.out_dim, self.x_dim ** 2)
        skip_connect = self.fc1(x).view(-1, self.out_dim, self.x_dim - 2, self.x_dim - 2)
        x = y + skip_connect
        y = self.ReLU(self.norm2(self.conv2(x)))
        x = x.view(-1, self.out_dim, (self.x_dim - 2) ** 2)
        skip_connect = self.fc2(x).view(-1, self.out_dim, self.x_dim - 4, self.x_dim - 4)
        x = y + skip_connect
        y = self.ReLU(self.norm3(self.conv3(x)))
        x = x.view(-1, self.out_dim, (self.x_dim - 4) ** 2)
        skip_connect = self.fc3(x).view(-1, self.out_dim, self.x_dim - 6, self.x_dim - 6)
        x = y + skip_connect
        return x

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = ResBlock(1, 64, 28)
        self.L2 = ResBlock(64, 128, 22)
        self.L3 = ResBlock(128, 256, 16)
        self.L4 = ResBlock(256, 512, 10)
        self.fc = nn.Linear(512 * 4 * 4, 10)

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.fc(x)
        return x

model = ResNet()

name = "ResNet"