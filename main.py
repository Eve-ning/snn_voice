#%%
import torch
from snntorch import spikegen
a = torch.Tensor([0, 0.5, 1])
#%%
spikegen.latency(a, num_steps=5, normalize=True, linear=False)

