import torch
import matplotlib.pyplot as plt
from IPython import display


def visualize_audio(wav: torch.Tensor, sr: int = 22050):
    # Average all channels
    if wav.dim() == 2:
        # Any to mono audio convertion
        wav = wav.mean(dim=0)

    plt.figure(figsize=(20, 5))
    plt.plot(wav, alpha=0.7, c="green")
    plt.grid()
    plt.xlabel("Time", size=20)
    plt.ylabel("Amplitude", size=20)
    plt.show()

    display.display(display.Audio(wav, rate=sr))
