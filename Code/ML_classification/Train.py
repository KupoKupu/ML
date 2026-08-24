import torch
from Data import train_dataloader, test_dataloader, test_data, device
# from MLP import model, name
# from CNN import model, name
# from ResNet import model, name
from UNet import model, name
# from RNN import model, name
# from Self_attention import model, name
# from Transformer import model, name
from pathlib import Path


print('Computing on', device)

model = model.to(device)
name = name

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))

def load():
    file_path = Path(name + '.pth')
    if file_path.is_file():
        cp = torch.load(name + '.pth')
        optimizer.load_state_dict(cp['optimizer' + name])
        model.load_state_dict(cp['model' + name])


def save():
    cp = {
        'model' + name: model.state_dict(),
        'optimizer' + name: optimizer.state_dict(),
    }
    torch.save(cp, name + '.pth')

EPOCHS = 200

def train(dataloader):
    for j in range(EPOCHS):
        running_loss = 0.
        i = 0
        print('Epoch', j + 1)
        for (x, y) in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.
            i += 1

def test(dataloader, data_size):
    num_correct = 0
    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        model.eval()
        output = model(x)
        output = output.tolist()
        y = y.tolist()
        for y1, label in zip(output, y):
            if y1.index(max(y1)) == label:
                num_correct += 1
    return num_correct / data_size

if __name__ == '__main__':
    model.train()
    load()
    train(train_dataloader)
    save()
    print('Done training!')

    model.eval()
    load()
    acc = test(test_dataloader, len(test_data))
    print('Test Accuracy:', acc)