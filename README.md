[![Python package](https://github.com/Eve-ning/snn_voice/actions/workflows/python-package.yml/badge.svg)](https://github.com/Eve-ning/snn_voice/actions/workflows/python-package.yml)

# Retraining SNN Conversions: CNN to SNN for Audio Classification Tasks

This is the repository for the paper titled in the heading, submitted for my NTU FYP

## Abstract
Efficient yet powerful models are in high demand for its portability and affordability.
Amongst other methods such as model-pruning, is limiting neural network operations to
sparse event-driven spikes: Spiking Neural Networks (SNNs) aims to unravel a new di-
rection in machine learning research. A significant amount of SNN literature straddles
upon mature works of artificial neural networks (ANNs) by migrating its architecture
and parameters into SNNs, optimizing the migration to retain as much performance as
possible. We spearhead a novel approach: the architecture is migrated and retrained
from scratch. We hypothesize that this new direction will unravel concepts that cur-
rently bottlenecks improvements in the field of SNN conversions. Furthermore, alike
Transfer Learning, inspire future efforts of fine-tuning a well converted model through
training.

This paper presents our analysis of training converted Convolutional Neural Networks
(CNNs) to SNNs on audio classification models. Results show that (1) SNN conver-
sions consistently underperforms CNNs marginally during training, however we also
show that model complexity has a possible association with this margin. (2) SNN con-
verts doesn’t necessarily approach the performance of its CNN counterparts asymptot-
ically by increasing the number of time-steps. (3) SNN training from scratch is costly
and impractical with current hardware and dedicated SNN optimization techniques are
necessary. (4) Enabling the SNN membrane decay rate to be learned doesn’t signifi-
cantly affect performance. This paper provides valuable insights into the perspective of
retraining converted SNNs for audio classification, and serves as a reference for future
studies and hardware implementation benchmarks.

