# train.py — real speech training loop

import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from models.speech_model import SpeechReconstructionModel
from preprocess.datasets import SpeechDataset
from config import DEVICE, BATCH_SIZE, LR, NUM_EPOCHS, SR


def collate_fn(batch):
    # pad to max length of longest waveform
    max_len = max(x.size(0) for x in batch)
    out = torch.stack([
        torch.nn.functional.pad(x, (0, max_len - x.size(0)))
        for x in batch
    ])
    return out


def train(dry_run=False):
    print("🚀 Loading dataset...")
    dataset = SpeechDataset(folder="data/train_wav", sr=SR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    device = DEVICE
    print(f"📌 Training on device: {device}")

    model = SpeechReconstructionModel(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(1 if dry_run else NUM_EPOCHS + 1):
        print(f"\n📚 Epoch {epoch}/{NUM_EPOCHS}")
        model.train()

        for i, batch in enumerate(loader):
            batch = batch.to(device)

            optimizer.zero_grad()
            output_audio = model(batch)

            # 🎯 target = original waveform (autoencoder-like training)
            loss = criterion(output_audio, batch)

            loss.backward()
            optimizer.step()

            print(f"  🟡 Batch {i+1}/{len(loader)} | Loss = {loss.item():.5f}")

            if dry_run:
                print("Dry run ended.")
                return

    print("🎉 Training complete! Saving model...")
    torch.save(model.state_dict(), "speech_model.pt")
    print("💾 Saved → speech_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    train(dry_run=args.dry_run)
