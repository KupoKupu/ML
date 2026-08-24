import torch
from torch import nn

class Attn(nn.Module):
    def __init__(self, embed_dim, key_dim, value_dim, seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.seq_len = seq_len
        self.key = nn.Linear(embed_dim, key_dim)
        self.query = nn.Linear(embed_dim, key_dim)
        self.value = nn.Linear(embed_dim, value_dim)
        self.SM = nn.Softmax(dim=2)

    def forward(self, x):
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        C = torch.matmul(self.SM(torch.matmul(Q, K.transpose(1, 2))), V)
        return C

class M_attn(Attn):
    def __init__(self, embed_dim, key_dim, value_dim, seq_len, num_heads):
        super().__init__(embed_dim, key_dim, value_dim, seq_len)
        self.num_heads = num_heads
        self.Attnlists = nn.ModuleList([Attn(embed_dim, key_dim, value_dim, seq_len) for i in range(num_heads)])
        self.Dense = nn.Linear(value_dim * num_heads, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        C = torch.cat([A(x.view(-1, self.seq_len, self.embed_dim)) for A in self.Attnlists], 2)
        C  = self.relu(self.Dense(C))
        return C

class Dense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x.reshape(-1, self.in_dim))


model = nn.Sequential(
    M_attn(28, 512, 512, 28, 16),
    Dense(28 * 28, 10))

name = "Self_attention"