import torch
from torch.nn.functional import pad


class CollateFn:
    def __init__(self, sr, classes):
        self.sr = sr
        self.classes = classes

    def __call__(self, batch):
        ars_audio = []
        ars_label = []

        for b in batch:
            if b[0].shape[-1] != self.sr:
                ar_audio = pad(b[0], [0, self.sr - b[0].shape[-1]], 'constant')
            else:
                ar_audio = b[0]
            ars_audio.append(ar_audio)
            ars_label.append(self.classes.index(b[2]))

        return torch.stack(ars_audio), torch.tensor(ars_label, dtype=torch.long)
