import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

from prediction.Classes import CLASSES

animaisTreino = os.listdir('data/treino')
animais = sorted(animal.split('.')[0] for animal in animaisTreino)
def adivinharDesenho(modelo,img):
    entrada = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.
    with torch.no_grad():
        out = modelo(entrada)
    animalPalpite = torch.argsort(-1 * out[0])
    palpites = [CLASSES[int(p.numpy())] for p in animalPalpite[:3]]

    print(palpites)
    probabilidades = torch.nn.functional.softmax(out[0], dim=0)
    valores, ids = torch.topk(probabilidades, 5)
    resultados = {animais[i]: v.item() for i, v in zip(ids, valores)}
    return resultados

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(5, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 16, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 36)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


modelo = Net()

state_dict = torch.load('./modelos/melhorModeloCom40Epocas2.pt', map_location='cpu')
modelo.load_state_dict(state_dict, strict=False)
modelo.eval()

pasta = f'data/imagens/fish_000000.png'

imagem = Image.open(pasta).resize((28,28)).convert('L')
array = np.array(imagem)

advinhacao = adivinharDesenho(modelo,array)
print(advinhacao)
