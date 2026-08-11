# optimizers.py
"""
WHAT THIS DOES:
Multi-class network (softmax + cross-entropy) trained with three
optimizers — plain gradient descent, momentum, and Adam — compared
head-to-head on the SAME architecture and data, isolating the
optimizer as the only variable.
"""
import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z: np.ndarray) -> np.ndarray:
    # Subtract the row-wise max before exponentiating — a real
    # numerical-stability trick: e^(large number) overflows, but
    # subtracting the max first keeps every exponent <= 0 without
    # changing the final softmax result (a property of the softmax
    # formula itself — shifting all inputs by a constant cancels out)
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    encoded = np.zeros((len(y), n_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded


class MultiClassNetwork:
    """
    Same 2-layer structure as yesterday, extended to K classes via
    softmax output + cross-entropy loss, with a pluggable optimizer.
    """

    def __init__(self, n_inputs: int, n_hidden: int, n_classes: int,
                 optimizer: str = "sgd", learning_rate: float = 0.1):
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(n_inputs, n_hidden) * np.sqrt(2.0 / n_inputs)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, n_classes) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_classes)
        self.lr = learning_rate
        self.optimizer = optimizer
        self.loss_history = []
        self._cache = {}

        # Optimizer state — momentum/Adam need to remember
        # information ACROSS steps, unlike plain SGD
        self.velocity = {"W1": 0, "b1": 0, "W2": 0, "b2": 0}       # momentum
        self.m = {"W1": 0, "b1": 0, "W2": 0, "b2": 0}              # Adam first moment
        self.v = {"W1": 0, "b1": 0, "W2": 0, "b2": 0}              # Adam second moment
        self.t = 0                                                  # Adam time step

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = softmax(z2)
        self._cache = {"X": X, "z1": z1, "a1": a1, "a2": a2}
        return a2

    def compute_gradients(self, y_onehot):
        X, z1, a1, a2 = (self._cache["X"], self._cache["z1"],
                          self._cache["a1"], self._cache["a2"])
        n = X.shape[0]

        # Softmax + cross-entropy's clean combined gradient —
        # exactly the same FORM as yesterday's sigmoid+log-loss case
        dz2 = a2 - y_onehot
        dW2 = (a1.T @ dz2) / n
        db2 = np.sum(dz2, axis=0) / n

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(z1)
        dW1 = (X.T @ dz1) / n
        db1 = np.sum(dz1, axis=0) / n

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def apply_update(self, grads):
        """Applies whichever optimizer this network was configured with."""
        if self.optimizer == "sgd":
            self._sgd_update(grads)
        elif self.optimizer == "momentum":
            self._momentum_update(grads)
        elif self.optimizer == "adam":
            self._adam_update(grads)

    def _sgd_update(self, grads):
        self.W1 -= self.lr * grads["W1"]
        self.b1 -= self.lr * grads["b1"]
        self.W2 -= self.lr * grads["W2"]
        self.b2 -= self.lr * grads["b2"]

    def _momentum_update(self, grads, beta=0.9):
        for name in ["W1", "b1", "W2", "b2"]:
            self.velocity[name] = beta * self.velocity[name] + (1 - beta) * grads[name]
            setattr(self, name, getattr(self, name) - self.lr * self.velocity[name])

    def _adam_update(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for name in ["W1", "b1", "W2", "b2"]:
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * grads[name]
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (grads[name] ** 2)

            # Bias correction: early in training, m and v are biased
            # toward zero (they started at zero and haven't accumulated
            # much yet) — this correction compensates for that,
            # a real, specific detail of why Adam has this exact form
            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)

            update = self.lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, getattr(self, name) - update)

    def compute_loss(self, y_onehot):
        a2 = self._cache["a2"]
        eps = 1e-15
        a2_clipped = np.clip(a2, eps, 1 - eps)
        return -np.mean(np.sum(y_onehot * np.log(a2_clipped), axis=1))

    def train(self, X, y, n_classes, n_epochs=300, verbose=False):
        y_onehot = one_hot(y, n_classes)
        for epoch in range(n_epochs):
            self.forward(X)
            loss = self.compute_loss(y_onehot)
            self.loss_history.append(loss)
            grads = self.compute_gradients(y_onehot)
            self.apply_update(grads)
            if verbose and epoch % 50 == 0:
                print(f"  [{self.optimizer}] epoch {epoch}: loss = {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)