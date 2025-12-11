# inference.py — load a wav, run model, save output
import argparse
import torch
from models.speech_model import SpeechReconstructionModel
from preprocess.audio_utils import load_wav, write_wav
from config import DEVICE

def run_inference(input_wav, output_wav):
    device = DEVICE
    model = SpeechReconstructionModel(device=device).to(device)
    model.eval()
    wav = load_wav(input_wav) # (time,)
    wav_t = wav.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(wav_t)
    # out is (batch, time) or (batch,1,time)
    audio = out.squeeze(0).cpu()
    write_wav(output_wav, audio)
    print('Saved', output_wav)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run_inference(args.input, args.output)