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
        C = torch.cat([A(x) for A in self.Attnlists], 2)
        C  = self.relu(self.Dense(C))
        return C

class Block(nn.Module):
    def __init__(self, embed_dim, key_dim, value_dim, seq_len, num_heads):
        super().__init__()
        self.Attn = M_attn(embed_dim, key_dim, value_dim, seq_len, num_heads)
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.dropout(self.Attn(self.ln(x)))
        return x

def PosEncode(len, dim):
    a = torch.zeros(len, dim)
    for i in range(len):
        for d in range(dim):
            if d % 2 == 0:
                a[i][d] = torch.sin(i / torch.pow(torch.tensor(10000.0), (d + 2) / dim))
            else:
                a[i][d] = torch.cos(i / torch.pow(torch.tensor(10000.0), (d + 1) / dim))
    return a

class ViT(nn.Module):
    def __init__(self, img_size, P_out, P_ker_size, P_stride, num_class, fc1_dim=64, attn_K_dim=128, attn_V_dim=128, attn_heads=3, layer_size=1):
        super().__init__()
        self.P_out = P_out
        self.Proj = nn.Conv2d(1, P_out, kernel_size=P_ker_size, stride=P_stride)
        self.fc1_dim = fc1_dim
        self.fc1 = nn.Linear(((img_size - P_ker_size) // P_stride + 1) ** 2, fc1_dim)
        self.Pos = nn.Parameter(PosEncode(P_out, fc1_dim))
        self.CLS = nn.Parameter(torch.zeros(1, 1, fc1_dim))
        self.Attn_layers = nn.Sequential(*[Block(fc1_dim, attn_K_dim, attn_V_dim, P_out + 1, attn_heads) for _ in range(layer_size)])
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.ln2 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, num_class)


    def forward(self, x):
        I = self.Proj(x).reshape(len(x), self.P_out, -1)
        I = self.fc1(I)
        I = self.ln1(I)
        I = I + self.Pos
        I = torch.cat((self.CLS.repeat(len(x), 1, 1), I), 1)
        I = self.Attn_layers(I)
        I = self.ln2(I[:, 0, :])
        I = self.fc2(I)
        return I

model = ViT(28, 256, 6, 1, 10, 256, 256, 256, 3, 3)

name = "ViT"