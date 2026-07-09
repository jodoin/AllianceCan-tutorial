"""The tutorial's one and only neural network.

A deliberately tiny 2-layer logistic regression:

    input (2 features) -> Linear -> ReLU -> Linear -> logits (2 classes)

The ML here is intentionally trivial. The whole point of this tutorial is to
teach *how to launch training jobs* on an Alliance Canada cluster (Narval),
not how to build fancy models. Keeping the model this small means every
example runs in seconds, on CPU or GPU, so you can focus on Slurm/Hydra/etc.
"""

import torch.nn as nn


class MoonsNet(nn.Module):
    """2-layer classifier for the two-class `make_moons` dataset.

    Parameters
    ----------
    in_features : int
        Number of input features (2 for make_moons).
    hidden : int
        Width of the hidden layer.
    num_classes : int
        Number of output classes (2 for make_moons).
    """

    def __init__(self, in_features: int = 2, hidden: int = 16, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        # Returns raw logits; cross-entropy loss applies softmax internally.
        return self.net(x)
