import torchaudio
from torch import nn


class MelFeaturizer(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=64,
        center=True,
    ):
        super(MelFeaturizer, self).__init__()

        self.featurizer = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=center,
        )

    def forward(self, wav, length=None):
        mel_spectrogram = self.featurizer(wav)
        mel_spectrogram = mel_spectrogram.clamp(min=1e-5).log()

        if length is not None:
            length = (length - self.featurizer.win_length) // self.featurizer.hop_length
            # We add `4` because in MelSpectrogram center==True
            length += 1 + 4

            return mel_spectrogram, length

        return mel_spectrogram
