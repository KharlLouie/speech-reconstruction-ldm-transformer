import torch
import torch.nn as nn

class HiFiGANWrapper(nn.Module):
    """
    Lightweight vocoder-style network.
    Acts as a placeholder for HiFi-GAN-style waveform synthesis.
    """

    def __init__(self, checkpoint_path=None, device="cpu"):
        super().__init__()
        self.device = device

        # Simple Conv1D-based vocoder (not full HiFi-GAN)
        self.net = nn.Sequential(
            nn.Conv1d(80, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 1, kernel_size=3, padding=1)
        )

    def forward(self, mel):
        """
        mel: (batch, mel_bins) or (batch, mel_bins, time)
        returns: waveform (batch, time)
        """

        if mel.dim() == 2:
            mel = mel.unsqueeze(-1)  # (batch, mel_bins, 1)

        audio = self.net(mel)       # (batch, 1, time)
        return audio.squeeze(1)
