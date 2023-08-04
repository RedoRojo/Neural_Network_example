import torch 
from torch import nn 
from torch.utils.data import DataLoader
from AcentosDataSet import AcentosDataset
import torchaudio
from NeuronalNetwork import FeedForwardNet

BATCH_SIZE = 128
EPOCHS = 1 
LEARNING_RATE = 0.001


def train_one_epoch(model, data_loader, loss_fn, optimiser, device): 
    # for each batch we want to calculate the loss and backpropagate the loss and the use gradiente descent the weights 
    for inputs, targets in data_loader: 
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss 
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # back propagate loss and update weights 
        optimiser.zero_grad() # reset the optimizer the grad  

        #calculate gradience
        loss.backward()
        optimiser.step() # update the weights 
    
    print(f"loss {loss.item()}")
    #printing the loss for the last batch 

def train(model, data_loader, loss_fn, optimiser, device, epochs):    
    # going throuth multiply epochs  

    for i in range(epochs): 
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("----------")
    
    print("Training is done")



if __name__ == "__main__": 

    ANNOTATIONS_FILE = "./annotations.csv"
    AUDIO_DIR = "./audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 110250 
    
    if torch.cuda.is_available(): 
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device {device}")

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
                         device = device)
    

    train_data_loader = DataLoader(ads, batch_size = BATCH_SIZE)

    cnn = FeedForwardNet().to(device) 

    # instantiate loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    #train model 
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    #save model
    torch.save(cnn.state_dict(), "audio.pth")

    print("Model trained and stored at audio.pth")

