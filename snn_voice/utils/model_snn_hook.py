from dataclasses import dataclass
from typing import Tuple

import torch

from snn_voice.utils.model_hook import ModelHook


@dataclass
class ModelSNNHook(ModelHook):
    def add_hook(self, key):
        """ Adds a post-forward hook to the model.
        Returns the hook for removal in the future.

        Example:
          h = add_hook(m3, "feature_extraction.conv1") will add hook to the submodule
          feature_extraction, and its submodule, conv1.

          By calling h.remove(), you will remove the hook added.

        Args:
          net: Model Instance to hook
          key: Name of Submodule
        """

        def hook(model, input, output):
            if self.hist.get(key, None) is None:
                self.hist[key] = []
            self.hist[key].append([o.detach() for o in output])
            return

        self.hooks.append(self.net.get_submodule(key).register_forward_hook(hook))

    def plot(self,
             input_ar: torch.Tensor,
             quantile_clamp: float = 0.85,
             resize: Tuple[int, int] = (50, 500),
             n_samples: int = 25, nrows=8):
        for k in list(self.hist.keys()):
            v = self.hist.pop(k)
            self.hist[f"{k}.spk"] = torch.concat([h[0] for h in v], dim=-2)
            self.hist[f"{k}.mem"] = torch.concat([h[1] for h in v], dim=-2)

        super().plot(input_ar, quantile_clamp, resize, n_samples, nrows)
