# Lightweight diffusion-style decoder placeholder.
# This is a simplified module to act as a latent-to-mel decoder.
# For research-quality LDM use AudioLDM or diffusion implementations.

import torch
import torch.nn as nn

class SimpleDiffusionDecoder(nn.Module):
    """
    Lightweight diffusion-inspired latent decoder.
    Acts as a latent-to-mel-spectrogram generator.
    """

    def __init__(self, input_dim=768, hidden_dim=512, mel_bins=80):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mel_bins),
        )

    def forward(self, latents):
        """
        latents: (batch, seq_len, dim) from Transformer encoder
        Returns: (batch, mel_bins)
        """

        # Pool across time dimension
        if latents.dim() == 3:
            x = latents.mean(dim=1)  # (batch, dim)
        else:
            x = latents

        mel = self.net(x)
        return mel
