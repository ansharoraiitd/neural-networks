# neuron.py
"""
WHAT THIS DOES:
Implements a single neuron and a small multi-layer network's
FORWARD PASS ONLY (no training yet — that's tomorrow). The goal
today is seeing exactly how numbers flow through a network, and
proving non-linearity's necessity directly, not just asserting it.
"""
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)  # same overflow-safety trick as Week 1 Saturday
    return 1 / (1 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def single_neuron_forward(X: np.ndarray, weights: np.ndarray, bias: float,
                           activation=sigmoid) -> np.ndarray:
    """
    One neuron. This IS logistic regression's forward pass —
    same equation as Week 1 Saturday, renamed.
    """
    z = X @ weights + bias
    return activation(z)


class TinyNetwork:
    """
    A minimal 2-layer network (1 hidden layer, 1 output neuron),
    forward pass only. Weights are random here — they'll be
    LEARNED via backprop tomorrow. Today just proves the
    architecture can represent non-linear boundaries even before
    any training happens, purely from having the right structure.
    """

    def __init__(self, n_inputs: int, n_hidden: int, use_activation: bool = True):
        # Small random init — not zero, because with multiple
        # neurons in a layer, identical starting weights would
        # make them learn identically forever (a real, named
        # problem called "symmetry" — random init breaks it)
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(n_inputs, n_hidden) * 0.5
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, 1) * 0.5
        self.b2 = np.zeros(1)
        self.use_activation = use_activation  # toggle to PROVE the linearity collapse

    def forward(self, X: np.ndarray) -> np.ndarray:
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1) if self.use_activation else z1  # <-- the entire experiment lives here
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)
        return a2