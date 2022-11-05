# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from snntorch import spikegen
from torchaudio import load
from torchaudio.transforms import MelSpectrogram

from src.dataset.sample_dataset import SampleDataset

ds = SampleDataset()
for i in ds.train_dl():
    print(i)
for i in ds.val_dl():
    print(i)
# for i in ds.test_dl():
#     print(i)
#%%

backward_path = Path("data/SpeechCommands/SpeechCommands/speech_commands_v0.02/backward/")
backward_fp_iter = backward_path.glob("*.wav")

fig, axs = plt.subplots(5, 1)
for ax in axs.flatten():
    ar, sr = load(next(backward_fp_iter).as_posix())

    mel_spectrogram = MelSpectrogram(
        sample_rate=sr,
        n_fft=500,
        center=True,
        pad_mode="reflect",
        power=2.0,
        normalized=True,
        norm="slaney",
        n_mels=16,
        mel_scale="htk",
    )
    ar_spec = mel_spectrogram(ar)
    ar_spec_log = torch.log(ar_spec)
    ar_spec_log = (ar_spec_log - ar_spec_log.min()) / (ar_spec_log.max() - ar_spec_log.min())

    ar_spike = spikegen.delta(ar_spec_log[0], threshold=0.01, padding=True)

    ax.imshow(ar_spike)
plt.show()
# spikegen.latency(a, num_steps=3, normalize=True, linear=True)
# %%
plt.imshow(np.log(ar_spec[0]))
plt.show()
