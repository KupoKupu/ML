import torch
from torch import nn
from Data import device

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(28, 256, batch_first=True)
        self.fc = nn.Linear(256, 10)
        self.key = nn.Linear(256, 128)
        self.query = nn.Linear(256, 128)
        self.SM = nn.Softmax(dim=2)

    def forward(self, x):
        h0 = torch.zeros(1, x.size()[0], 256).to(device)
        x, h = self.rnn(x.view(-1, 28, 28), h0)
        h = h.view(-1, 1, 256)
        K = self.key(x)
        Q = self.query(h)
        C = torch.matmul(self.SM(torch.matmul(Q, K.transpose(1, 2))), x)
        C = C.view(-1, 256)
        C = self.fc(C)
        return C

model = RNN()

name = "RNN"
