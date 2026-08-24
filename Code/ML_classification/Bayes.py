import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=(lambda x: torch.floor(ToTensor()(x) * 255 / 128))
)

test = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=(lambda x: torch.floor(ToTensor()(x) * 255 / 128))
)


def len_labels(l):
    i = torch.tensor(0.0)
    for samples in train:
        if l == samples[1]:
            i += 1
    return i

Log_P_label = torch.Tensor([torch.log(len_labels(l) / len(train)) for l in range(10)])

P_matrix = torch.zeros(10, 28, 28)

for samples in train:
    P_matrix[samples[1]] += samples[0][0]


for l in range(10):
    P_matrix[l] = P_matrix[l] / len_labels(l)
    print('Done P_matrix:', l)


Log_P_matrix = torch.zeros(10, 28, 28)
Log_comp_P_matrix = torch.zeros(10, 28, 28)

non_zero_idx = (P_matrix != 0.0)
Log_P_matrix[non_zero_idx] = torch.log(P_matrix[non_zero_idx])
Log_comp_P_matrix[non_zero_idx] = torch.log(1 - P_matrix[non_zero_idx])

def predict(x):
    return ((Log_P_matrix * x + Log_comp_P_matrix * (1 - x)).reshape(10, -1).sum(dim=1) + Log_P_label).argmax()

i = 0

for item in test:
    if predict(item[0][0]) == item[1]:
        i += 1

print('Test Accuracy:', i / len(test))