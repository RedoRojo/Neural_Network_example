import os
import torch 
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

class AcentosDataset(Dataset):
    def __init__(self,
                 device, 
                 audio_dir, 
                 annotation_file, 
                 transformation,
                 target_sample_rate,
                 num_samples
                ):
        self.device = device 
        self.annotations = pd.read_csv(annotation_file) 
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
        self.transformation = transformation.to(self.device)
    

    def __len__(self): 
        num_files  = len([f for f in os.listdir(self.audio_dir) 
                           if os.path.isfile(os.path.join(self.audio_dir, f))
                        ])
        return num_files
    

    def __getitem__(self, index): 
        audio_path = self._get_audio_path(index)
        label = self._get_audio_label(index)
        signal, sr = torchaudio.load(audio_path)
        signal = signal.to(self.device)

        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label 

    def _cut_if_necessary(self, signal): 
        # signal -> Tensor -> (number of channels, num_samples)
        if signal.shape[1] > self.num_samples: 
            signal = signal[:, :self.num_samples]
        return signal 

    def _right_pad_if_necessary(self, signal): 
        length_signal = signal.shape[1]
        if length_signal < self.num_samples: 
            # [1,1,1] - > [1,1,1,0,0]
            # append values to the the array 
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples) # (0,2 )
            # [1,1,1] - > [1,1,1,0,0]
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate: 
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal 

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1: 
            signal = torch.mean(signal, dim = 0, keepdim=True) 
        return signal

    def _get_audio_path(self, index):

        fold = self.audio_dir
        path = os.path.join(fold, self.annotations.iloc[index, 0])
        return path 

    def _get_audio_label(self, index): 
        return self.annotations.iloc[index, 1]