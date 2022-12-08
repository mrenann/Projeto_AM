import torch
from flask import Flask, render_template, request
from torch import nn
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import re
import io
import base64

from prediction.Classes import CLASSES

app = Flask(__name__,
            template_folder='web/templates',
            static_url_path='',
            static_folder='web/static')
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1152, 512),
    nn.ReLU(),
    nn.Linear(512, 36),
)
state_dict = torch.load('./prediction/modelos/melhorModeloCom40Epocas.pt', map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()

@app.route('/',methods=['GET','POST'])
def root():
    if request.method == 'GET':
        return render_template('desenho.html')
    else:
        imagemB64 = request.values['imageBase64']
        imageData = re.sub('^data:image/.+;base64,', '', imagemB64)
        imageData = base64.b64decode(imageData)
        with Image.open(io.BytesIO(imageData)) as im:
                imagem = im.resize((28, 28))
                back = Image.new("RGB", imagem.size, (255, 255, 255))
                back.paste(imagem, mask=imagem.split()[3])
                back.save("desenho.jpg", "JPEG", quality=100)

                jpg = Image.open("desenho.jpg").convert('L')
                imagemInvertida = ImageOps.invert(jpg)
                img = np.array(imagemInvertida)
                plt.imshow(img)
                plt.savefig('av.png')
                x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.
                with torch.no_grad():
                    out = model(x)
                animalPalpite = torch.argsort(-1 * out[0])
                palpites = [CLASSES[int(p.numpy())] for p in animalPalpite[:3]]

                print(palpites)
        predicao = {
            'predicao': palpites[0],
        }

        return predicao



@app.errorhandler(404)
def pageNotFound(error):
    return render_template('notfound.html')


if __name__ == '__main__':
    app.run()
