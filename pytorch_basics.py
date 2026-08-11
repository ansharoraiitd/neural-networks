# pytorch_basics.py
"""
WHAT THIS DOES:
Part 1: rebuild Wednesday's from-scratch network in PyTorch,
confirming autograd produces the same training behavior as your
hand-coded backprop. Part 2: a small CNN for real image data.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class DenseNet(nn.Module):
    """
    Same architecture as Wednesday's MultiClassNetwork — 2 layers,
    ReLU hidden, raw logits out (softmax handled by the loss
    function, see explanation above).
    """
    def __init__(self, n_inputs=64, n_hidden=32, n_classes=10):
        super().__init__()
        self.layer1 = nn.Linear(n_inputs, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)


class SimpleCNN(nn.Module):
    """
    A small CNN: 2 convolutional blocks (conv -> relu -> pool),
    then flatten into a fully-connected classifier head.
    Sized for 28x28 grayscale images (Fashion-MNIST).
    """
    def __init__(self, n_classes=10):
        super().__init__()
        # in_channels=1 (grayscale), out_channels=16 (16 different
        # learned filters, each producing its own feature map),
        # kernel_size=3, padding=1 keeps spatial size unchanged
        # going INTO the pool step
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves spatial size each call

        # After 2 pool operations: 28 -> 14 -> 7, with 32 channels
        # from conv2 -> flattened feature count = 32 * 7 * 7
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # 28x28 -> 14x14, 16 channels
        x = self.pool(torch.relu(self.conv2(x)))   # 14x14 -> 7x7,  32 channels
        x = x.view(x.size(0), -1)                   # flatten, keeping batch dim
        x = torch.relu(self.fc1(x))
        return self.fc2(x)                          # raw logits, same reasoning as DenseNet


def train_pytorch_model(model, X_train, y_train, n_epochs=300, lr=0.01, verbose=False):
    """
    A standard PyTorch training loop. Compare this structure
    directly against Wednesday's train() method — same steps,
    same order, autograd handling what backward()/compute_gradients()
    did manually before.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # expects raw logits, applies softmax internally
    loss_history = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()               # clear gradients from the PREVIOUS step —
                                              # PyTorch accumulates by default, must reset
        outputs = model(X_train)             # forward pass — builds the computation graph
        loss = criterion(outputs, y_train)
        loss.backward()                      # autograd walks the graph backward — THIS
                                              # is the automated version of Tuesday's
                                              # hand-coded backward() method
        optimizer.step()                     # apply the update — same role as your
                                              # _adam_update() from Wednesday

        loss_history.append(loss.item())
        if verbose and epoch % 50 == 0:
            print(f"  epoch {epoch}: loss = {loss.item():.4f}")

    return loss_history