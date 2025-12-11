# Minimal HiFi-GAN wrapper (expects mel spectrogram input)
# NOTE: This is a wrapper placeholder. You need a real HiFi-GAN checkpoint for production.

import torch
import torch.nn as nn

class HiFiGANWrapper(nn.Module):
    def __init__(self, checkpoint_path=None, device="cpu"):
        super().__init__()
        self.device = device
        # placeholder simple invert network (NOT real HiFi-GAN)
        self.net = nn.Sequential(
            nn.Conv1d(80, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 1, kernel_size=3, padding=1)
        )


def forward(self, mel):
    # mel: (batch, mel_bins) or (batch, mel_bins, time)
    if mel.dim() == 2:
        # fake time axis
        mel = mel.unsqueeze(-1) # (batch, mel_bins, 1)
    audio = self.net(mel)
    # audio: (batch, 1, time)
    return audio.squeeze(1)