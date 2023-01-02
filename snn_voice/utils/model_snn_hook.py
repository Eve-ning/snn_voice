from dataclasses import dataclass

import torch

from snn_voice.utils.model_hook import ModelHook


@dataclass
class ModelSNNHook(ModelHook):
    def _add_hook(self, key):
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

        return self.net.get_submodule(key).register_forward_hook(hook)

    def forward(self, input_ar):
        self.hist = {}

        hooks = self._add_hooks()
        self.net(input_ar)

        for k in list(self.hist.keys()):
            v = self.hist.pop(k)
            self.hist[f"{k}.spk"] = torch.concat([h[0] for h in v], dim=-2)
            self.hist[f"{k}.mem"] = torch.concat([h[1] for h in v], dim=-2)

        self._remove_hooks(hooks)

        return input_ar[:, 0].permute(1, 0, 2)
