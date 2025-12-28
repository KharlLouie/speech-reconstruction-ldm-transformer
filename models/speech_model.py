import torch
import torch.nn as nn
from .transformer_encoder import TransformerEncoder
from .diffusion_decoder import SimpleDiffusionDecoder
from .vocoder import HiFiGANWrapper

class SpeechReconstructionModel(nn.Module):
    def __init__(self, encoder_name="facebook/wav2vec2-base", device="cpu"):
        super().__init__()
        self.encoder = TransformerEncoder(model_name=encoder_name)
        # wav2vec2-base hidden dim = 768
        self.decoder = SimpleDiffusionDecoder(
            input_dim=768,
            hidden_dim=512,
            mel_bins=80
        )
        self.vocoder = HiFiGANWrapper(device=device)

    def forward(self, waveforms, return_mel=False):
        """
        waveforms: (batch, time)
        """
        latents = self.encoder(waveforms)      # (B, T, 768)
        mel = self.decoder(latents)             # (B, 80)
        audio = self.vocoder(mel)               # (B, time)

        if return_mel:
            return audio, mel

        return audio
