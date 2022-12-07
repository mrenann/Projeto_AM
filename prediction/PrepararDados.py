import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

base = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
pasta = r'./data/'
pastaPlots = r'./plots/'
pastaModels = r'./modelos/'
pastaTreino = os.path.join(pasta, 'treino')
pastaValidacao = os.path.join(pasta, 'validacao')
pastaTeste = os.path.join(pasta, 'teste')
with open('categorias/categoriasAnimais.txt', 'r') as f:
    categorias = [line.strip() for line in f]

print("Preparando Dados ...")


def baixar():
    for categoria in categorias:
        troca_ = categoria.replace('_', '%20')
        endereco = base + troca_ + '.npy'
        print('Baixando ' + categoria + '...')
        urllib.request.urlretrieve(endereco, pasta + categoria + '.npy')


def salvar(listaAnimais):
    for animal in listaAnimais:
        if animal == "teste" or animal == "treino" or animal == "validacao":
            continue
        print("Salvando " + animal + "...")
        dados = np.load(pasta + animal)
        np.random.shuffle(dados)
        dados = dados[:35000]
        teste, validacao, treino = np.split(dados, [int(0.2 * len(dados)), int(0.44 * len(dados))])
        np.save(pastaTeste + '/' + animal, teste)
        np.save(pastaValidacao + '/' + animal, validacao)
        np.save(pastaTreino + '/' + animal, treino)
        os.remove(pasta + animal)


def verificarPastas():
    if not os.path.isdir(pastaPlots):
        os.makedirs(pastaPlots)
    if not os.path.isdir(pastaModels):
        os.makedirs(pastaModels)
    if not os.path.isdir(pastaTeste):
        os.makedirs(pastaTeste)
    if not os.path.isdir(pastaTreino):
        os.makedirs(pastaTreino)
    if not os.path.isdir(pastaValidacao):
        os.makedirs(pastaValidacao)


if not os.path.isdir(pasta):
    os.makedirs(pasta)
    baixar()
    verificarPastas()
    print(" ~ Dados Baixados!")
else:
    print(" ~ Dados já Baixados")

if os.path.isdir(pasta) and len(os.listdir(pastaTeste)) == 0:
    salvar(os.listdir(pasta))
    print(" ~ Dados Salvos!")
else:
    print(" ~ Dados já Salvos")

leao = np.load(pastaTreino + '/lion.npy')
plt.figure(figsize=(2, 2))
plt.imshow(leao[3].reshape(28, 28), cmap="gray")
plt.savefig("plots/leao.png")

print("Fim da preparação de Dados :) \n\n")
