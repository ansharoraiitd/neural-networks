# day4_week4_pytorch_cnn.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from pytorch_basics import DenseNet, SimpleCNN, train_pytorch_model

# ------------------------------------------------------------------
# PART 1: Confirm PyTorch autograd matches your hand-coded network
# on the SAME data from Wednesday
# ------------------------------------------------------------------
print("=" * 60)
print("PART 1: PyTorch autograd vs Tuesday/Wednesday's hand-coded backprop")
print("=" * 60)
data = load_digits()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors — this is the ONLY structurally new step;
# everything after this mirrors Wednesday's train() logic
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

dense_model = DenseNet(n_inputs=64, n_hidden=32, n_classes=10)
loss_history = train_pytorch_model(dense_model, X_train_t, y_train_t, n_epochs=300, lr=0.01, verbose=True)

with torch.no_grad():  # no need to track gradients for evaluation — saves memory/compute
    test_preds = torch.argmax(dense_model(X_test_t), dim=1)
    test_acc = (test_preds == y_test_t).float().mean().item()
print(f"\nPyTorch DenseNet test accuracy: {test_acc:.4f}")
print("(Compare this against Wednesday's Adam result on the same dataset — "
      "should land in a similar range, confirming autograd is doing the same job)")

# ------------------------------------------------------------------
# PART 2: Real CNN on real image data — Fashion-MNIST
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2: CNN on Fashion-MNIST")
print("=" * 60)

transform = transforms.Compose([transforms.ToTensor()])  # scales pixels to [0,1] automatically
train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# Use a subset for today's runtime — full 60k images works fine
# too, just slower; today's goal is confirming the mechanism, not
# squeezing out maximum accuracy
train_subset = torch.utils.data.Subset(train_dataset, range(6000))
test_subset = torch.utils.data.Subset(test_dataset, range(1000))

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

cnn_model = SimpleCNN(n_classes=10)
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

n_epochs = 5
for epoch in range(n_epochs):
    cnn_model.train()  # sets the model to training mode (matters for
                        # layers like dropout/batchnorm — not used
                        # today, but this is the standard convention)
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"  Epoch {epoch+1}/{n_epochs}: avg loss = {avg_loss:.4f}")

# Evaluate
cnn_model.eval()  # sets to evaluation mode — pairs with .train() above
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        preds = torch.argmax(cnn_model(images), dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"\nCNN test accuracy on Fashion-MNIST subset: {correct/total:.4f}")

torch.save(cnn_model.state_dict(), "cnn_fashion_mnist.pth")
print("Saved cnn_fashion_mnist.pth — the trained model weights, "
      "for tomorrow's evaluation and error analysis")