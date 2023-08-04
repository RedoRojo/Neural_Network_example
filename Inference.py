import torch 
import torchaudio
from NeuronalNetwork import CNNNetwork 
from AcentosDataSet import AcentosDataset
from Train import ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES



class_mapping = [
    # traduccion de las etiquetas 
    "camba",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

def predict(model, input, target, class_mapping): 
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



if __name__ == "__main__":

    # load back the model
    cnn = CNNNetwork()

    state_dict = torch.load("audio.pth")

    cnn.load_state_dict(state_dict)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        hop_length=512,
        n_mels=64
    )
    ads = AcentosDataset(annotation_file = ANNOTATIONS_FILE, 
                         audio_dir = AUDIO_DIR, 
                         transformation = mel_spectrogram, 
                         target_sample_rate = SAMPLE_RATE, 
                         num_samples = NUM_SAMPLES, 
                         device = "cpu")

    #get sample 
    input, target = ads[0][0], ads[0][1]

    input.unsqueeze_(0)

    #make an inference
    predicted, expected = predict(cnn, input, target, class_mapping)

    print(f"Predicted: '{predicted}', expected: '{expected}'")