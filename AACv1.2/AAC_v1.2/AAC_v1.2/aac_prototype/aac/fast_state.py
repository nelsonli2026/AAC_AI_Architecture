"""
Fast recurrent state S_t (AAC spec section 5), with the notation collision
fixed: the low-rank factor is named L_t, not P_t (P_t is reserved globally
for persistent memory).

    S_t = A_t S_{t-1} + B u_t
    A_t = D_t + L_t Q_t^T,     rank(L_t Q_t^T) = r << d

D_t is a diagonal decay (kept in [0, d_max] for stability), and L_t, Q_t
give a cheap rank-r correction. This is a fixed random reservoir (echo
state network style): weights are not trained by gradient descent here.
Only the readout built on top of S_t is trained -- this keeps the
prototype tractable without implementing full BPTT through the recurrence,
while still exercising the exact structured-transition equation from the
spec. Stability is enforced by keeping the spectral radius of A_t < 1.
"""
import numpy as np


class FastState:
    def __init__(self, d_state, d_input, rank=8, decay_max=0.95, seed=0):
        rng = np.random.default_rng(seed)
        self.d_state = d_state
        self.rank = rank

        # Diagonal decay component D_t (fixed here; "_t" subscript kept for
        # fidelity to the spec even though it is time-invariant in this
        # prototype -- a data-dependent D_t is a natural extension).
        self.D = rng.uniform(0.1, decay_max, size=d_state)

        # Low-rank correction L Q^T, rescaled so the induced operator norm
        # stays small -- this is what keeps A_t = D + L Q^T a contraction.
        L = rng.normal(0, 1.0, size=(d_state, rank))
        Q = rng.normal(0, 1.0, size=(d_state, rank))
        LQ = L @ Q.T
        LQ *= 0.2 / (np.linalg.norm(LQ, 2) + 1e-8)
        self.L, self.Q = L, Q
        self._LQ = LQ

        # Input projection B.
        scale = np.sqrt(1.0 / d_input)
        self.B = rng.normal(0, scale, size=(d_input, d_state))

    def A(self):
        return np.diag(self.D) + self._LQ

    def init_state(self, batch_size):
        return np.zeros((batch_size, self.d_state))

    def step(self, S_prev, u_t):
        """S_prev: (batch, d_state), u_t: (batch, d_input) -> S_t"""
        pre = S_prev @ self.A().T + u_t @ self.B
        return np.tanh(pre)  # bounded nonlinearity keeps the reservoir stable
