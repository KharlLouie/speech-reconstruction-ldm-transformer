# Transformer encoder wrapper using Wav2Vec2 from Hugging Face
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Config


class TransformerEncoder(torch.nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)


    def forward(self, waveforms, attention_mask=None):
        # waveforms: (batch, time)
        outputs = self.model(waveforms, attention_mask=attention_mask)
        # last_hidden_state: (batch, seq_len, hidden_dim)
        return outputs.last_hidden_state

