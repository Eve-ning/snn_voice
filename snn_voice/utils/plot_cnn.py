from dataclasses import dataclass, field
from typing import Tuple

import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import Resize, InterpolationMode


@dataclass
class PlotCNN:
    net: nn.Module
    hooks: list = field(default_factory=lambda: [
        "conv_blks.conv_blk1",
        "conv_blks.conv_blk2",
        "conv_blks.conv_blk3",
        "conv_blks.conv_blk4",
    ])
    hist: dict = field(init=False, default_factory=dict)

    def forward(self, input_ar):
        self.hist = {}

        hooks = self._add_hooks()
        self.net(input_ar)
        self._remove_hooks(hooks)
        return input_ar

    def plot(self,
             input_ar: torch.Tensor,
             quantile_clamp: float = 0.85,
             resize: Tuple[int, int] = (50, 500),
             n_samples: int = 25):
        """ Plots the hist

        Args:
            input_ar: Input Array to plot alongside the hist plot.
            quantile_clamp: The quantile to clamp the values at.
            resize: THe final size of each hist plot
            n_samples: Limit the number of samples (from 1 batch) to plot
        """

        input_ar = self.forward(input_ar)
        rs = Resize(resize, interpolation=InterpolationMode.NEAREST)

        def clamp(ar):
            ar = torch.abs(ar)
            return torch.clamp(ar, 0, torch.quantile(ar, quantile_clamp))

        def make_hist_grid(x):
            """ Makes the grid image from PyTorch-like B x H x W tensors.

            Notes:
              In the context of SpeechCommands, we have B x F x T. Thus, we also expand
              on the 2nd axes to mimic a grayscale channel.
            """
            x = clamp(x)
            x = (x - x.min()) / (x.max() - x.min())
            return rs(x).permute(1, 2, 0)

        fig, axs = plt.subplots(1 + len(self.hist), )

        hist = {'input': input_ar, **self.hist}
        for ax, (hist_name, ar_hist) in zip(axs.flatten(), hist.items()):
            ax.set_title(hist_name)
            ax.axis('off')
            im = make_hist_grid(ar_hist[:n_samples])
            ax.imshow(im)
            fig.tight_layout()

        self.hist = {}

    def _add_hook(self, key):
        """ Adds a post-forward hook to the model.
        Returns the hook for removal in the future.

        Example:
            h = add_hook(m3, "feature_extraction.conv1") will add hook to the submodule
            feature_extraction, and its submodule, conv1.

        Args:
            net: Model Instance to hook
            key: Name of Submodule
        """

        def hook(model, input, output):
            if type(output) == tuple:
                if self.hist.get(key, None) is None:
                    self.hist[key] = []
                self.hist[key].append([o.detach() for o in output])
                return

            self.hist[key] = output.detach()

        return self.net.get_submodule(key).register_forward_hook(hook)

    def _add_hooks(self):
        return [self._add_hook(hook) for hook in self.hooks]

    @staticmethod
    def _remove_hooks(hooks):
        for hook in hooks:
            hook.remove()
