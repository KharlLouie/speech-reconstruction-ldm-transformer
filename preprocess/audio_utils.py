import torchaudio
import torch

from config import SR

def load_wav(path, sr=SR):
    wav, orig_sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True) # mono
    if orig_sr != sr:
        wav = torchaudio.transforms.Resample(orig_sr, sr)(wav)
    return wav.squeeze(0)

def write_wav(path, waveform, sr=SR):
    torchaudio.save(path, waveform.unsqueeze(0), sr)