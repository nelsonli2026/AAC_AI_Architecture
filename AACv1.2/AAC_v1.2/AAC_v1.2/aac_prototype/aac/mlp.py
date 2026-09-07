"""
A tiny, dependency-free MLP with hand-derived forward/backward passes and
Adam. No autodiff framework is used anywhere in this prototype (no PyTorch
is available in this environment) -- gradients below are worked out by hand
for a standard 2-layer MLP with ReLU hidden units and a linear or softmax
output.

This is the only piece of "learning by gradient descent" in the prototype.
The memory system itself (write/promote/evict/merge/reinforce) learns via
the explicit online update rules given in the AAC spec, not backprop --
that is faithful to the original document, where those are described as
algorithmic decisions driven by a learned utility *score*, not parameters
trained end-to-end through the whole recurrent system.
"""
import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class MLP:
    """Linear -> ReLU -> Linear, with either a sigmoid or softmax head."""

    def __init__(self, d_in, d_hidden, d_out, head="softmax", seed=0):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / d_in)
        scale2 = np.sqrt(2.0 / d_hidden)
        self.params = {
            "W1": rng.normal(0, scale1, size=(d_in, d_hidden)),
            "b1": np.zeros(d_hidden),
            "W2": rng.normal(0, scale2, size=(d_hidden, d_out)),
            "b2": np.zeros(d_out),
        }
        assert head in ("softmax", "sigmoid", "linear")
        self.head = head
        self.opt = Adam(self.params, lr=2e-3)
        self._cache = None

    def forward(self, x, train=False):
        """x: (batch, d_in) -> y: (batch, d_out)"""
        W1, b1, W2, b2 = (self.params[k] for k in ("W1", "b1", "W2", "b2"))
        z1 = x @ W1 + b1
        h = np.maximum(z1, 0.0)  # ReLU
        z2 = h @ W2 + b2
        if self.head == "softmax":
            z2 = z2 - z2.max(axis=-1, keepdims=True)
            e = np.exp(z2)
            y = e / e.sum(axis=-1, keepdims=True)
        elif self.head == "sigmoid":
            y = 1.0 / (1.0 + np.exp(-z2))
        else:
            y = z2
        if train:
            self._cache = (x, z1, h, y)
        return y

    def backward_softmax_xent(self, targets_idx):
        """Cross-entropy grad for softmax head. targets_idx: (batch,) int labels."""
        x, z1, h, y = self._cache
        batch = x.shape[0]
        dz2 = y.copy()
        dz2[np.arange(batch), targets_idx] -= 1.0
        dz2 /= batch
        return self._backward_common(x, z1, h, dz2)

    def backward_sigmoid_bce(self, targets):
        """BCE grad for sigmoid head. targets: (batch, d_out) in [0,1]."""
        x, z1, h, y = self._cache
        batch = x.shape[0]
        dz2 = (y - targets) / batch
        return self._backward_common(x, z1, h, dz2)

    def _backward_common(self, x, z1, h, dz2):
        W2 = self.params["W2"]
        dW2 = h.T @ dz2
        db2 = dz2.sum(axis=0)
        dh = dz2 @ W2.T
        dz1 = dh * (z1 > 0)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        self.opt.step(self.params, grads)
        return grads

    def loss_softmax_xent(self, y, targets_idx):
        batch = y.shape[0]
        p = np.clip(y[np.arange(batch), targets_idx], 1e-9, 1.0)
        return -np.log(p).mean()
