# day2_week4_backprop.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from backprop import TrainableNetwork, numerical_gradient_check


def plot_boundary(network, X, y, ax, title=""):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = network.forward(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm", levels=20)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=20)
    ax.set_title(title)


X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------------
# PART 1: Gradient checking — verify the backprop math BEFORE
# trusting it to actually train anything
# ------------------------------------------------------------------
print("=" * 60)
print("PART 1: NUMERICAL GRADIENT CHECK")
print("=" * 60)
net = TrainableNetwork(n_inputs=2, n_hidden=8, learning_rate=0.5)
numerical_gradient_check(net, X_train, y_train, param_name="W1")
print("\n(Compare these numbers by eye against dW1 if you print it inside "
      "backward() — they should be very close. This confirms the chain-rule "
      "derivation above was implemented correctly.)")

# ------------------------------------------------------------------
# PART 2: Actually train it
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2: TRAINING")
print("=" * 60)
net = TrainableNetwork(n_inputs=2, n_hidden=8, learning_rate=0.5)
net.train(X_train, y_train, n_epochs=1000)

train_acc = np.mean(net.predict(X_train).ravel() == y_train)
test_acc = np.mean(net.predict(X_test).ravel() == y_test)
print(f"\nFinal train accuracy: {train_acc:.4f}")
print(f"Final test accuracy: {test_acc:.4f}")

# ------------------------------------------------------------------
# PART 3: Before vs after — the actual payoff of today
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

untrained_net = TrainableNetwork(n_inputs=2, n_hidden=8)
plot_boundary(untrained_net, X, y, axes[0], "BEFORE training (random weights)")
plot_boundary(net, X, y, axes[1], f"AFTER training (test acc={test_acc:.3f})")

plt.tight_layout()
plt.savefig("before_after_training.png")
print("\nSaved before_after_training.png")

# Loss curve
plt.figure(figsize=(8, 5))
plt.plot(net.loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss (log-loss)")
plt.title("Training Loss Over Time")
plt.savefig("training_loss_curve.png")
print("Saved training_loss_curve.png")