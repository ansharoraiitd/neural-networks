# day1_week4_forward_pass.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from neuron import TinyNetwork


def plot_boundary(model, X, y, ax, title=""):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm", levels=20)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=20)
    ax.set_title(title)


# make_moons: two interleaving crescents — NOT linearly separable,
# same spirit as Wednesday Week 2's concentric circles for SVM
X, y = make_moons(n_samples=300, noise=0.15, random_state=42)

print("=" * 60)
print("UNTRAINED networks — random weights, forward pass only")
print("=" * 60)
print("Neither network has been trained yet. Today only tests")
print("whether the ARCHITECTURE can even represent a curved")
print("boundary in principle — training happens tomorrow.\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# No activation: layers collapse to one linear transformation,
# no matter how many "layers" exist on paper
net_no_activation = TinyNetwork(n_inputs=2, n_hidden=8, use_activation=False)
plot_boundary(net_no_activation, X, y, axes[0], "NO activation (collapses to linear)")

# With ReLU: genuinely non-linear, can bend around the moons shape
net_with_activation = TinyNetwork(n_inputs=2, n_hidden=8, use_activation=True)
plot_boundary(net_with_activation, X, y, axes[1], "WITH ReLU activation")

plt.tight_layout()
plt.savefig("linearity_collapse_proof.png")
print("Saved linearity_collapse_proof.png")

# Confirm the collapse ALGEBRAICALLY too, not just visually:
# a "2-layer" network with no activation should be exactly
# equivalent to ONE linear layer with combined weights
print("=" * 60)
print("ALGEBRAIC PROOF: combining two linear layers = one linear layer")
print("=" * 60)
combined_W = net_no_activation.W1 @ net_no_activation.W2
combined_b = net_no_activation.b1 @ net_no_activation.W2 + net_no_activation.b2
X_sample = X[:3]
via_two_layers = X_sample @ net_no_activation.W1 @ net_no_activation.W2 + combined_b
via_one_layer = X_sample @ combined_W + combined_b
print("Output via the 'two-layer' path:\n", via_two_layers.ravel())
print("Output via a single combined linear layer:\n", via_one_layer.ravel())
print(f"Identical (within floating point): {np.allclose(via_two_layers, via_one_layer)}")