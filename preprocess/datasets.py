# preprocess/datasets.py

import os
import torch
from torch.utils.data import Dataset
from .audio_utils import load_wav

class SpeechDataset(Dataset):
    def __init__(self, folder, sr):
        self.folder = folder
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".wav")
        ]
        self.sr = sr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav = load_wav(self.files[idx], sr=self.sr)
        return torch.tensor(wav, dtype=torch.float32)
