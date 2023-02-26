from torchaudio_augmentations import *

AUDIO_PATH = "data/ESC-50-master/audio/"
SR = 16000

TRANSFORM_SET_1 = Compose(
    transforms=[
        RandomApply([PolarityInversion()], p=0.8),
        RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
        RandomApply([Gain()], p=0.2),
        HighLowPass(sample_rate=SR * 5),
        RandomApply([Delay(sample_rate=SR * 5)], p=0.5),
        RandomApply([PitchShift(n_samples=SR * 5, sample_rate=SR)], p=0.4),
        RandomApply([Reverb(sample_rate=SR)], p=0.3),
    ]
)
