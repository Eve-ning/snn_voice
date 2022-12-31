from dataclasses import dataclass, field

from torch import nn


@dataclass
class ModelHook:
    net: nn.Module
    hist: dict = field(default_factory=dict)
    hooks: list = field(default_factory=list)

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
            self.hist[key] = output.detach()

        self.hooks.append(self.net.get_submodule(key).register_forward_hook(hook))

    def remove_all_hooks(self):
        for h in self.hooks:
            h.remove()

        self.hooks = []
