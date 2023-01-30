import torch
from snntorch.spikegen import latency, rate


def latency_replica(x, n_steps: int) -> torch.Tensor:
    x = x[:, 0].abs()

    # TODO: Unsure why x is unnormalized? May be of concern
    x = (x - x.min()) / (x.max() - x.min())

    return latency(x, num_steps=n_steps, clip=True, threshold=0.01, normalize=True).unsqueeze(2)


def rate_replica(x, n_steps: int) -> torch.Tensor:
    x = x[:, 0].abs()
    x = (x - x.min()) / (x.max() - x.min())
    return rate(x, num_steps=n_steps).unsqueeze(2)


def repeat_replica(x, n_steps: int) -> torch.Tensor:
    # This dynamically repeats on a new axis (at the front).
    # E.g., if x.shape = (A, B, C)
    #   x.repeat(n_steps, 1, 1, 1) <- the number of 1s are dynamically yield from the shape
    #   The number of 1s is dynamic as x has varying ndims due to non-spec and spectrogram transforms
    return x.repeat(n_steps, *(1,) * x.ndim)
