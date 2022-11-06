from __future__ import annotations

from typing import Tuple

import torch
from torchaudio import load

from src.settings import DATA_SAMPLE_DIR


def load_any_sample(ix: int = 0, sample_name: str = "") -> Tuple[torch.Tensor, int] | None:
    """ Loads any sample from the data sample dataset

    Args:
        ix: Index of the sample
        sample_name: Word Utterance name. Must match sample folder name, else returns None

    Returns:
        If found, a Torch Tensor & sample rate tuple. Else None.

    """
    if sample_name:
        sample_name += "/"
    else:
        sample_name = "**/"
    for e, i in enumerate(DATA_SAMPLE_DIR.glob(f"{sample_name}*.wav")):
        if e != ix:
            continue
        return load(i.as_posix())
    return None
