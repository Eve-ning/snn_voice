from dataclasses import dataclass, field
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import Resize, InterpolationMode
from torchvision.utils import make_grid


@dataclass
class PlotHist:
    net: nn.Module
    hist: dict = field(init=False, default_factory=dict)
    im_height: int = 250
    im_width: int = 500
    im_padding: int = 3
    im_padding_value: int = 1
    hook_target: str = None

    def __post_init__(self):
        self.net_name = self.net.__class__.__name__

        # dynamically determine the hook target
        if self.hook_target is None:
            self.hook_target = 'cnn' if 'CNN' in self.net_name else 'snn'

        self.hooks = [
            f'{self.hook_target}.{e}'
            for e, _ in enumerate(self.net.get_submodule(self.hook_target))
        ]

    def forward(self, input_ar):
        self.hist = {}

        hooks = self._add_hooks()
        self.net(input_ar)
        self._remove_hooks(hooks)

    def plot(self, input_ar: torch.Tensor, subplots_kwargs: dict = {}):
        """ Plots the network history given the input array

        Args:
            input_ar: Input Array to plot alongside the hist plot.
            subplots_kwargs: Additional KWArgs for plt.subplots
        """

        # Propagate array through network
        self.forward(input_ar)

        ims = {}
        for k, ar in self.hist.items():
            # If SNN Piczak Input: [TS x BS x FB x 1 x TB]
            # If CNN Piczak Input: [1  x BS x FB x 1 x TB]
            # If SNN M5     Input: [TS x BS x FB x TB]
            # If CNN M5     Input: [1  x BS x FB x TB]

            # Normalize the history shape
            # We'll pull TS x FB x TB
            ar = torch.stack(ar)
            if self.net_name.startswith("Piczak"):
                im_t = ar[:, 0, :, 0, :]
            elif self.net_name.startswith("M5"):
                im_t = ar[:, 0, :, :]
            elif self.net_name.startswith("Hjh"):
                im_t = ar[:, 0, :, 0, :]

            # Make grid from history
            # If there's n_steps, then it'll be stacked on the x-axis
            im_t = im_t.unsqueeze(1)  # Add temp channel

            time_steps = im_t.shape[0]
            rs = Resize(
                (self.im_height, int(self.im_height / time_steps)),
                interpolation=InterpolationMode.NEAREST
            )

            # We resize EACH time step
            im_t = rs(im_t)

            im_t = im_t.abs()
            im = make_grid(
                im_t,
                nrow=25,
                padding=self.im_padding,
                pad_value=self.im_padding_value,
                normalize=True
            )
            im = im.permute(1, 2, 0)
            ims[k] = im

        fig, axs = plt.subplots(2, int(len(ims) / 2), **subplots_kwargs)

        for (k, im), ax in zip(ims.items(), axs.flatten()):
            ax.imshow(im)
            ax.set_title(k)
            ax.axis('off')
        fig.suptitle(
            f"{self.net.__class__.__name__}'s "
            f"History. "
            f"{datetime.now().isoformat()}")
        plt.tight_layout()
        plt.show()

        self.hist = {}

    def _add_hook(self, key):
        """ Adds a post-forward hook to the model.
        Returns the hook for removal in the future.

        Example:
            h = add_hook(m3, "feature_extraction.conv1") will add hook to the submodule
            feature_extraction, and its submodule, conv1.

        Args:
            key: Name of Submodule
        """

        def hook(model, input, output):
            if self.hist.get(key, None) is None:
                self.hist[key] = []
            self.hist[key].append(output.detach())

        return self.net.get_submodule(key).register_forward_hook(hook)

    def _add_hooks(self):
        return [self._add_hook(hook) for hook in self.hooks]

    @staticmethod
    def _remove_hooks(hooks):
        for hook in hooks:
            hook.remove()
