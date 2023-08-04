from flask import Flask, request, Response
from NeuronalNetwork import FeedForwardNet
import jsonpickle
import torch 
import torchaudio
import os
from AcentosDataSet import AcentosDataset

import numpy as np

app = Flask(__name__)

class_mapping = [
    # traduccion de las etiquetas 
    "camba",
    "1",
    "2",
    "3",
]

ANNOTATIONS_FILE = "./annotations.csv"
AUDIO_DIR = "./audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 110250 
    
def cut_if_necessary(signal): 
    # signal -> Tensor -> (number of channels, num_samples)
    if signal.shape[1] > NUM_SAMPLES: 
        signal = signal[:, :NUM_SAMPLES]
    return signal 

def right_pad_if_necessary(signal): 
    length_signal = signal.shape[1]
    if length_signal < NUM_SAMPLES: 
        # [1,1,1] - > [1,1,1,0,0]
        # append values to the the array 
        num_missing_samples = NUM_SAMPLES - length_signal
        last_dim_padding = (0, num_missing_samples) # (0,2 )
        # [1,1,1] - > [1,1,1,0,0]
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    
    return signal

def resample_if_necessary(signal, sr):
    if sr != SAMPLE_RATE: 
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE).to('cpu')
        signal = resampler(signal)
    return signal 

def mix_down_if_necessary(signal):
    if signal.shape[0] > 1: 
        signal = torch.mean(signal, dim = 0, keepdim=True) 
    return signal


def predict(model, input, class_mapping, target=0): 
    model.eval()
    # some layer are turned off 
    # model.train() this turns on the things again 

    with torch.no_grad(): 
        predictions = model(input)
        # tensor (1, 10) one sample, 10 classes or categories 
        # (1, 10) = [0.3232, 0.032, ...]
        predicted_index = predictions[0].argmax(0) #obtain the index of the greatest values
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    return predicted, expected

red = FeedForwardNet()
state_dict = torch.load("acentos.pth")
red.load_state_dict(state_dict)

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft = 1024,
    hop_length=512,
    n_mels=64
)

@app.route('/api/audio', methods=['POST'])
def test():
    re = request
    # recibir y guardar audio
    audio_file = re.files['audio']
    filename = audio_file.filename
    file_path = os.path.join(os.path.dirname(__file__), filename)
    audio_file.save(file_path)

    # obtener tensor para prediccion
    signal, sr = torchaudio.load(file_path)
    signal = signal.to("cpu")
    signal = resample_if_necessary(signal, sr)
    signal = mix_down_if_necessary(signal)
    signal = right_pad_if_necessary(signal)
    signal = cut_if_necessary(signal)
    signal = mel_spectrogram(signal)

   
    respuesta, _ = predict(model=red, input = signal, class_mapping = class_mapping)
    
    response = {'mensaje' : "Se reconoce el acento " + str(respuesta) }
    response_pickled = jsonpickle.encode(response)
    return Response(response= response_pickled, status = 200, mimetype="application/json")
    
app.run()