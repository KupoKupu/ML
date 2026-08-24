import torch
from torchvision import transforms
from PIL import Image
import os
from Data import device
from Train import model


directory = os.path.expanduser("~\\PycharmProjects\\HandSamples")

for filename in sorted(os.listdir(directory)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img = Image.open(os.path.join(directory, filename)).convert('L')
        image = transforms.ToTensor()(img)
        image = torch.unsqueeze(image.to(device), 0)
        print(filename)
        m = model(image)[0]
        print(torch.argmax(m))