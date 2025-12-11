# train.py — skeleton training loop
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from models.speech_model import SpeechReconstructionModel
from config import DEVICE, BATCH_SIZE, LR, NUM_EPOCHS
from preprocess.audio_utils import load_wav

class DummyDataset(Dataset):
    def __init__(self, n=16, length=SR*2):
        self.n = n
        self.length = length

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.randn(self.length)
        return x

def collate_fn(batch):
    # pad to max length
    max_len = max(x.size(0) for x in batch)
    out = torch.stack([torch.nn.functional.pad(x, (0, max_len - x.size(0))) for x in batch])
    return out

def train(dry_run=False):
    device = DEVICE
    model = SpeechReconstructionModel(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    dataset = DummyDataset(n=16, length=SR*2)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for epoch in range(1 if dry_run else NUM_EPOCHS):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            # create dummy target waveform (same shape)
            target = torch.randn_like(out)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            print('loss', loss.item())
            if dry_run:
                return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    train(dry_run=args.dry_run)