import torchaudio
from torch import nn
from typing import Optional


class SpectrogramFeaturizer(nn.Module):
    def forward(self, wav, length=None):
        spectrogram = self.featurizer(wav)
        spectrogram = spectrogram.clamp(min=1e-5).log()

        if length is not None:
            length = (length - self.featurizer.win_length) // self.featurizer.hop_length
            # We add `4` because in MelSpectrogram center==True
            length += 1 + 4

            return spectrogram, length

        return spectrogram


class MelFeaturizer(SpectrogramFeaturizer):
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


class MFCCFeaturizer(SpectrogramFeaturizer):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        dct_type: int = 2,
        norm: str = "ortho",
        log_mels: bool = False,
        melkwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.featurizer = torchaudio.transforms.MFCC(
            sample_rate, n_mfcc, dct_type, norm, log_mels, melkwargs=melkwargs
        )
