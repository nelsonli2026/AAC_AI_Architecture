"""
A minimal reverse-mode autodiff engine.
"""
import numpy as np


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op", "probs", "_id")
    _counter = 0

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad or any(
            getattr(c, "requires_grad", False) for c in _children
        )
        if self.requires_grad and self.grad is None:
            self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        # Stable, content-independent iteration order. `_prev` used to be
        # a plain `set(_children)`; Tensor defines no custom __hash__ or
        # __eq__, so that fell back to Python's identity-based default,
        # and a set's iteration order over identity-hashed objects depends
        # on their memory addresses -- not fixed by any RNG seed, and not
        # guaranteed stable across separate process runs. Since floating-
        # point addition isn't exactly associative, that silently changed
        # the order gradients from multiple parents accumulate into a
        # shared child, and that bit-level noise compounded over thousands
        # of training episodes into visibly different trained models even
        # with every explicit seed in the codebase held fixed (see
        # PHASE3_GATE_RESULTS.md, Methods note item 3, for the rerun that
        # exposed this). Fixed by tagging every Tensor with a creation-
        # order id and deduping+ordering children by that id instead of
        # by set membership -- deterministic regardless of memory layout,
        # and `_prev` is never used for anything but iteration elsewhere
        # in this file, so a list is a safe drop-in replacement for a set.
        Tensor._counter += 1
        self._id = Tensor._counter
        seen = {}
        for c in _children:
            seen[id(c)] = c
        self._prev = sorted(seen.values(), key=lambda t: t._id)
        self._op = _op
        self.probs = None

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self.grad is not None:
            self.grad[...] = 0.0

    def backward(self):
        topo = _topo_sort(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

        # --- break reference cycles ---------------------------------
        # Every op below creates `out` and then does `out._backward =
        # _backward`, where `_backward` is a closure that captures `out`
        # itself (to read out.grad). That's a direct reference cycle on
        # every non-leaf Tensor node in the graph -- potentially thousands
        # per episode. Plain refcounting can't free a cycle; only the
        # generational cyclic GC can, and it gets steadily more expensive
        # as cyclic garbage accumulates across many episodes, which shows
        # up as training getting slower and slower over a run (not a
        # correctness bug, but a severe and worsening performance one).
        # Clearing _backward on every non-leaf node here breaks the cycle
        # so ordinary refcounting frees the whole per-episode graph the
        # moment the caller's local variables (h, M, k, v, logits, ...)
        # go out of scope -- no waiting on the cyclic collector at all.
        # Leaf/parameter tensors (created with the default _op="") are
        # left untouched since they persist across episodes and must keep
        # working as graph roots in every subsequent forward pass.
        for v in topo:
            if v._op != "":
                v._backward = None


def _topo_sort(root):
    """Iterative post-order traversal (children fully processed before
    their parent is appended) -- functionally identical to the natural
    recursive version:

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)

    but implemented with an explicit stack instead of the Python call
    stack. A long episode (hundreds of tokens, each chaining several ops
    through a per-step recurrent state/memory) builds a dependency chain
    thousands of nodes deep. The recursive version needs one Python stack
    frame per node in that chain and blows through Python's default
    recursion limit (1000) well before it blows through any real memory
    limit -- this shows up as a RecursionError specifically on long
    sequences (e.g. the gap=300-400 generalization sweep), even though
    shorter training-length episodes never hit it. The iterative version
    has no such ceiling; it's bounded by available memory, not call-stack
    depth."""
    topo = []
    visited = {id(root)}
    stack = [(root, iter(root._prev))]
    while stack:
        node, it = stack[-1]
        nxt = next(it, None)
        if nxt is None:
            topo.append(node)
            stack.pop()
        elif id(nxt) not in visited:
            visited.add(id(nxt))
            stack.append((nxt, iter(nxt._prev)))
    return topo


def add(a, b):
    out = Tensor(a.data + b.data, _children=(a, b), _op="add")

    def _backward():
        if a.requires_grad:
            a.grad += out.grad
        if b.requires_grad:
            b.grad += out.grad
    out._backward = _backward
    return out


def sub(a, b):
    out = Tensor(a.data - b.data, _children=(a, b), _op="sub")

    def _backward():
        if a.requires_grad:
            a.grad += out.grad
        if b.requires_grad:
            b.grad -= out.grad
    out._backward = _backward
    return out


def mul_const(a, c):
    out = Tensor(a.data * c, _children=(a,), _op="mul_const")

    def _backward():
        if a.requires_grad:
            a.grad += out.grad * c
    out._backward = _backward
    return out


def scale(vec, scalar_t):
    s = float(np.asarray(scalar_t.data).reshape(()) if scalar_t.data.ndim == 0
              else scalar_t.data.reshape(-1)[0])
    out = Tensor(vec.data * s, _children=(vec, scalar_t), _op="scale")

    def _backward():
        if vec.requires_grad:
            vec.grad += out.grad * s
        if scalar_t.requires_grad:
            scalar_t.grad += np.sum(out.grad * vec.data)
    out._backward = _backward
    return out


def tanh(x):
    y = np.tanh(x.data)
    out = Tensor(y, _children=(x,), _op="tanh")

    def _backward():
        if x.requires_grad:
            x.grad += out.grad * (1.0 - y * y)
    out._backward = _backward
    return out


def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x.data))
    out = Tensor(y, _children=(x,), _op="sigmoid")

    def _backward():
        if x.requires_grad:
            x.grad += out.grad * y * (1.0 - y)
    out._backward = _backward
    return out


def relu(x):
    y = np.maximum(x.data, 0.0)
    out = Tensor(y, _children=(x,), _op="relu")

    def _backward():
        if x.requires_grad:
            x.grad += out.grad * (x.data > 0)
    out._backward = _backward
    return out


def matvec(W, x):
    out = Tensor(W.data @ x.data, _children=(W, x), _op="matvec")

    def _backward():
        if W.requires_grad:
            W.grad += np.outer(out.grad, x.data)
        if x.requires_grad:
            x.grad += W.data.T @ out.grad
    out._backward = _backward
    return out


def vecmat(x, W):
    out = Tensor(x.data @ W.data, _children=(x, W), _op="vecmat")

    def _backward():
        if x.requires_grad:
            x.grad += W.data @ out.grad
        if W.requires_grad:
            W.grad += np.outer(x.data, out.grad)
    out._backward = _backward
    return out


def outer(a, b):
    out = Tensor(np.outer(a.data, b.data), _children=(a, b), _op="outer")

    def _backward():
        if a.requires_grad:
            a.grad += out.grad @ b.data
        if b.requires_grad:
            b.grad += out.grad.T @ a.data
    out._backward = _backward
    return out


def dot(a, b):
    out = Tensor(np.dot(a.data, b.data), _children=(a, b), _op="dot")

    def _backward():
        if a.requires_grad:
            a.grad += out.grad * b.data
        if b.requires_grad:
            b.grad += out.grad * a.data
    out._backward = _backward
    return out


def stack(vec_list):
    vecs = list(vec_list)
    out = Tensor(np.stack([v.data for v in vecs]), _children=tuple(vecs),
                 _op="stack")

    def _backward():
        for i, v in enumerate(vecs):
            if v.requires_grad:
                v.grad += out.grad[i]
    out._backward = _backward
    return out


def concat(a, b):
    n = a.data.shape[0]
    out = Tensor(np.concatenate([a.data, b.data]), _children=(a, b), _op="concat")

    def _backward():
        if a.requires_grad:
            a.grad += out.grad[:n]
        if b.requires_grad:
            b.grad += out.grad[n:]
    out._backward = _backward
    return out


def embedding_lookup(table, idx):
    out = Tensor(table.data[idx].copy(), _children=(table,), _op="embed")

    def _backward():
        if table.requires_grad:
            table.grad[idx] += out.grad
    out._backward = _backward
    return out


def softmax(x):
    z = x.data - x.data.max()
    e = np.exp(z)
    y = e / e.sum()
    out = Tensor(y, _children=(x,), _op="softmax")

    def _backward():
        if x.requires_grad:
            s = np.sum(out.grad * y)
            x.grad += y * (out.grad - s)
    out._backward = _backward
    return out


def softmax_cross_entropy(logits, label_idx):
    z = logits.data - logits.data.max()
    e = np.exp(z)
    p = e / e.sum()
    loss_val = -np.log(np.clip(p[label_idx], 1e-12, 1.0))
    out = Tensor(loss_val, _children=(logits,), _op="softmax_xent")
    out.probs = p

    def _backward():
        if logits.requires_grad:
            d = p.copy()
            d[label_idx] -= 1.0
            logits.grad += out.grad * d
    out._backward = _backward
    return out


def bce_loss(p, y, pos_weight=1.0):
    """Binary cross-entropy for a scalar/1-elem Tensor p (already a
    probability, e.g. sigmoid output) against a python float label y in
    {0,1}. pos_weight upweights the positive class to counter imbalance
    (rare 'write here' positions vs. abundant filler positions):
    -[pos_weight*y*log(p) + (1-y)*log(1-p)]."""
    pv = float(np.asarray(p.data).reshape(-1)[0])
    pv_c = np.clip(pv, 1e-9, 1.0 - 1e-9)
    loss_val = -(pos_weight * y * np.log(pv_c) + (1 - y) * np.log(1 - pv_c))
    out = Tensor(np.asarray(loss_val).reshape(()), _children=(p,), _op="bce")

    def _backward():
        if p.requires_grad:
            dp = (-pos_weight * y / pv_c + (1 - y) / (1 - pv_c))
            p.grad += out.grad * dp
    out._backward = _backward
    return out


def free_graph(root):
    """Break the self-referencing backward closures on every non-leaf
    node reachable from `root`, WITHOUT computing any gradients.

    Tensor.backward() does this same cycle-breaking, but only as a side
    effect of actually running backprop. Eval-mode forward passes
    (train=False) never call .backward() at all -- there's nothing to
    differentiate -- so every eval call was leaking its entire per-call
    graph as uncollectable cyclic garbage, same root cause as the
    training-time bug, just untouched by that fix since it lives inside
    a method that eval code path never calls.

    Call this on the final loss/output Tensor of any forward pass that
    won't have .backward() called on it (i.e. whenever train=False).
    Uses the same iterative (non-recursive) traversal as backward() --
    see _topo_sort's docstring for why that matters on long episodes."""
    topo = _topo_sort(root)
    for v in topo:
        if v._op != "":
            v._backward = None
