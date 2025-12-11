# simple mel extraction script
import argparse
import torch
import librosa
import numpy as np
from config import SR, N_MELS, HOP_LENGTH, N_FFT
import soundfile as sf

def wav_to_mel(path, out_path=None):
    y, sr = librosa.load(path, sr=SR)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if out_path:
        np.save(out_path, mel_db)
    return mel_db

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    m = wav_to_mel(args.input, args.output)
    print('mel shape', m.shape)