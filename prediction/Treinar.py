# Imports Necessários

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import TensorDataset
from torchvision import transforms

from utils.MetricTools import MetricTools
from utils.PlotTools import PlotTools

# Inicio Código #

# Transformando as imagens
transformacoesImagens = {
    'treino': transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ]),
    'validacao': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'teste': transforms.Compose([
        transforms.ToTensor(),
    ])
}

# Carregando as imagens e determinando as pastas
pasta = r'./data/'
pastaTreino = os.path.join(pasta, 'treino') + '/'
pastaValidacao = os.path.join(pasta, 'validacao') + '/'
pastaTeste = os.path.join(pasta, 'teste') + '/'

pastas = {
    'treino': pastaTreino,
    'validacao': pastaValidacao,
    'teste': pastaTeste,
}

# Tamanho do batch de treinamento
batch = 128

# Numero de classes
numeroClasses = len(os.listdir(pastas['treino']))
print("Número de classes : " + str(numeroClasses) + " Classes")
# Nome das classes
animaisTreino = os.listdir(pastas['treino'])
animais = sorted(animal.split('.')[0] for animal in animaisTreino)
animalIndice = {animais[i]: i for i in range(len(animais))}
indiceAnimal = {i : animais[i] for i in range(len(animais))}
print('Animais:', animais)

# Criando o dataSet
datasets = {}
print('Criando o dataset...')
for dataset in ['treino', 'validacao', 'teste']:
    dataX = []
    dataY = []
    for caminhoPasta, _, animaisPasta in os.walk(pastas[dataset]):
        for animalPasta in animaisPasta:
            animal = animalPasta.split('.')[0]
            x = np.load(caminhoPasta + animalPasta).reshape(-1, 28, 28) / 255
            y = np.ones((len(x), 1), dtype=np.int64) * animalIndice[animal]

            dataX.extend(x)
            dataY.extend(y)
    datasets[dataset] = torch.utils.data.TensorDataset(
        torch.stack([transformacoesImagens[dataset](Image.fromarray(np.uint8(i * 255))) for i in dataX]),
        torch.stack([torch.Tensor(j) for j in dataY]))

numImagensTreino = len(datasets['treino'])
numImagensValidacao = len(datasets['validacao'])

dataloaders = {
    dataset: torch.utils.data.DataLoader(datasets[dataset], batch_size=batch, shuffle=True) for dataset in
    ['treino', 'validacao', 'teste']
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
            nn.Linear(100, numeroClasses)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


modeloTreinado = Net()

modeloTreinado.to(device)

metricaErro = nn.CrossEntropyLoss()

otimizador = optim.Adam(modeloTreinado.parameters(), lr=0.001)


def treinar(dataloaders, modelo, otimizador, metricaErro, device, caminhoModelo, numeroEpocas):
    print('Começando o treinamento do modelo...')
    minimoErroValidacao = np.Inf
    historicoErroTreino = []
    historicoErroValidacao = []

    for epoca in range(1, numeroEpocas + 1):
        acuraciaTreino = 0.0
        acuraciaValidacao = 0.0
        erroTreino = 0.0
        erroValidacao = 0.0

        modelo.train()
        for i, (entradas, labels) in enumerate(dataloaders['treino']):
            entradas = entradas.to(device)
            labels = labels.long().to(device)

            otimizador.zero_grad()
            saidas = modelo(entradas)
            erro = metricaErro(saidas, torch.max(labels, 1)[0])

            erro.backward()
            otimizador.step()
            erroTreino += ((1 / (i + 1)) * (erro.data - erroTreino)).cpu()
            valoresMaximos, indicesValoresMaximos = torch.max(saidas.data, 1)
            predicoesCorretas = indicesValoresMaximos.eq(labels.data.view_as(indicesValoresMaximos))
            acuracia = torch.mean(predicoesCorretas.type(torch.FloatTensor))
            acuraciaTreino += acuracia.item() * entradas.size(0)

        with torch.no_grad():
            modelo.eval()
            for i, (entradas, labels) in enumerate(dataloaders['validacao']):
                entradas = entradas.to(device)
                labels = labels.long().to(device)

                saidas = modelo(entradas)
                erro = metricaErro(saidas, torch.max(labels, 1)[0])

                erroValidacao += ((1 / (i + 1)) * (erro.data - erroValidacao)).cpu()

                valoresMaximos, indicesValoresMaximos = torch.max(saidas.data, 1)
                predicoesCorretas = indicesValoresMaximos.eq(labels.data.view_as(indicesValoresMaximos))
                acuracia = torch.mean(predicoesCorretas.type(torch.FloatTensor))
                acuraciaValidacao += acuracia.item() * entradas.size(0)

        mediaAcuraciaTreino = acuraciaTreino / numImagensTreino
        mediaAcuraciaValidacao = acuraciaValidacao / numImagensValidacao

        print(
            "Época : {:03d} \n\t\tTreino - Erro: {:.4f}, Acurácia: {:.4f}%, \n\t\tValidação - Erro : {:.4f}, Acurácia: {:.4f}%".format(
                epoca, erroTreino, mediaAcuraciaTreino * 100, erroValidacao, mediaAcuraciaValidacao * 100))

        historicoErroTreino.append(erroTreino)
        historicoErroValidacao.append(erroValidacao)

        if erroValidacao < minimoErroValidacao:
            print('Salvando Modelo...')
            minimoErroValidacao = erroValidacao
            torch.save(modelo.state_dict(), caminhoModelo)

    return modelo.cpu(), historicoErroTreino, historicoErroValidacao


# train the model
epocas = 40
modeloTreinado, historicoErroTreino, historicoErroValidacao = treinar(dataloaders, modeloTreinado, otimizador,
                                                                      metricaErro, device,
                                                                      'modelos/melhorModeloCom' + str(
                                                                          epocas) + 'Epocas.pt', epocas)

plt.plot(historicoErroTreino, label='Treinamento')
plt.plot(historicoErroValidacao, label='Validação')
plt.title('Erros Treinamento e Validação')
plt.xticks([i for i in range(0, len(historicoErroTreino), 5)].append(len(historicoErroValidacao)))
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.legend()
_ = plt.ylim()
plt.savefig('plots/treinamentoCom' + str(epocas) + 'Epocas.png')
plt.show()

modeloTreinado.load_state_dict(torch.load('modelos/melhorModeloCom' + str(epocas) + 'Epocas.pt'))


def testar(loaders, modelo, metricaErro, device):
    y = None
    yHat = None
    erroTreino = 0.0
    acertados = 0.0
    total = 0.0

    modelo.eval()
    for i, (entradas, labels) in enumerate(loaders['teste']):
        entradas = entradas.to(device)
        labels = labels.long().to(device)

        saidas = modelo(entradas)
        erro = metricaErro(saidas, labels.view(-1))
        erroTreino += ((1 / (i + 1)) * (erro.data - erroTreino))
        predicao = saidas.data.max(1, keepdim=True)[1]

        if y is None:
            y = labels.cpu().numpy()
            yHat = predicao.data.cpu().view_as(labels).numpy()
        else:
            y = np.append(y, labels.cpu().numpy())
            yHat = np.append(yHat, predicao.data.cpu().view_as(labels).numpy())

        acertados += np.sum(np.squeeze(predicao.eq(labels.data.view_as(predicao))).cpu().numpy())
        total += entradas.size(0)

    print("Teste \n\t\tErro: {:.4f}".format(erroTreino))
    print("Acurácia: %2d%% (%2d / %2d)" % (100. * acertados / total, acertados, total))

    return y, yHat


y, yHat = testar(dataloaders, modeloTreinado, metricaErro, "cpu")
MetricTools.accuracy(y, yHat)
matriz = MetricTools.confusionMatrix(y, yHat, nclasses=len(animais))

PlotTools.confusionMatrix(matriz, animais, title='Matriz de Confusão',
                          filename='matrizConfusaoCom' + str(epocas) + 'Epocas', figsize=(20, 20), path='plots/')