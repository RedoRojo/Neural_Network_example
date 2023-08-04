# import torchaudio

# signal, sr = torchaudio.load("./audio/audio_voz_0.mp3")



# print(type(signal))
# print(signal.shape)
# print(signal)
# print(type(sr))
# print(sr)

from AcentosDataSet import AcentosDataset
import torch
import torchaudio

if __name__ == "__main__":
    ANNOTATIONS_FILE = "./annotations.csv"
    AUDIO_DIR = "./audio"
    SAMPLE_RATE = 11025
    NUM_SAMPLES = 55125 
    
    if torch.cuda.is_available(): 
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        hop_length=512,
        n_mels=20
    )
    usd = AcentosDataset(annotation_file = ANNOTATIONS_FILE, 
                         audio_dir = AUDIO_DIR, 
                         transformation = mel_spectrogram, 
                         target_sample_rate = SAMPLE_RATE, 
                         num_samples = NUM_SAMPLES, 
                         device = device)
    
    print(usd[10][0].shape)

