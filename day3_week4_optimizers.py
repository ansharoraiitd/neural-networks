# day3_week4_optimizers.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from optimizers import MultiClassNetwork

# load_digits: 8x8 images of handwritten digits 0-9, flattened to
# 64 features — real multi-class data (10 classes), small enough
# to train quickly from scratch, sets up Thursday's CNN nicely
# since it's the same KIND of data (images) at a small scale
data = load_digits()
X, y = data.data, data.target
n_classes = 10

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Digits dataset: {X_train.shape[0]} train, {X_test.shape[0]} test, "
      f"{n_classes} classes, {X_train.shape[1]} features (8x8 pixels flattened)")

results = {}
for optimizer_name in ["sgd", "momentum", "adam"]:
    print(f"\n{'=' * 60}")
    print(f"TRAINING WITH: {optimizer_name.upper()}")
    print("=" * 60)
    net = MultiClassNetwork(
        n_inputs=64, n_hidden=32, n_classes=n_classes,
        optimizer=optimizer_name, learning_rate=0.01 if optimizer_name == "adam" else 0.5
    )
    net.train(X_train_scaled, y_train, n_classes=n_classes, n_epochs=300, verbose=True)

    test_acc = np.mean(net.predict(X_test_scaled) == y_test)
    print(f"  Final test accuracy: {test_acc:.4f}")
    results[optimizer_name] = {"loss_history": net.loss_history, "test_acc": test_acc}

# Plot all three loss curves together — the actual comparison
plt.figure(figsize=(9, 5))
for name, data_dict in results.items():
    plt.plot(data_dict["loss_history"], label=f"{name} (test acc={data_dict['test_acc']:.3f})")
plt.xlabel("Epoch")
plt.ylabel("Loss (cross-entropy)")
plt.title("Optimizer Comparison: SGD vs Momentum vs Adam")
plt.legend()
plt.savefig("optimizer_comparison.png")
print("\nSaved optimizer_comparison.png")