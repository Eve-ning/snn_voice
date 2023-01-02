from dataclasses import dataclass

from torchvision.utils import make_grid

from snn_voice.utils.plot_cnn import PlotCNN


@dataclass
class PlotSNN(PlotCNN):
    padding_f: float = 0.1

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
            padding = int(v[0][0].shape[1] * self.padding_f)
            self.hist[f"{k}_spk"] = make_grid([h[0] for h in v], nrow=1, padding=padding)[:1]
            self.hist[f"{k}_mem"] = make_grid([h[1] for h in v], nrow=1, padding=padding)[:1]

        self._remove_hooks(hooks)

        return input_ar[:, 0].permute(1, 0, 2)
