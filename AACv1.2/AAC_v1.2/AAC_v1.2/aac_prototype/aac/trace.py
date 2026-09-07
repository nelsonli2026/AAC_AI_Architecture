"""
Temporary trace T_t (AAC spec section 9): a small buffer where candidate
(key, value) events accumulate evidence before being trusted enough to
promote into persistent memory. Implements the observation/commitment
separation that is the spec's central mechanism:

    S_t -> T_t -> P_t,   with progressively longer retention.

Each slot decays every step (lambda_t); repeated evidence for the same key
increases its promotion score. This is a simple, literal reading of
T_t = lambda_t T_{t-1} + F_T(x_t, S_t): F_T here is "add/refresh a slot".
"""
import numpy as np


class TemporaryTrace:
    def __init__(self, d_key, d_val, capacity=32, decay=0.85,
                 promote_threshold=0.6, merge_cos_threshold=0.95, seed=0):
        self.d_key, self.d_val = d_key, d_val
        self.capacity = capacity
        self.decay = decay
        self.promote_threshold = promote_threshold
        self.merge_cos_threshold = merge_cos_threshold

        self.K = np.zeros((0, d_key))
        self.V = np.zeros((0, d_val))
        self.strength = np.zeros((0,))  # accumulated evidence / promote score

    def step_decay(self):
        if len(self.K) == 0:
            return
        self.strength *= self.decay
        keep = self.strength > 1e-3
        self.K, self.V, self.strength = self.K[keep], self.V[keep], self.strength[keep]

    def add_candidate(self, key, value, utility_score):
        """utility_score in [0,1] from the learned utility estimator; this
        is the F_T(x_t, S_t) write into the trace, gated by value-of-
        information rather than accepting every event unconditionally."""
        if len(self.K) > 0:
            sims = PersistentMemoryLikeCosSim(key, self.K)
            best = int(np.argmax(sims)) if len(sims) else -1
            if best >= 0 and sims[best] >= self.merge_cos_threshold:
                self.strength[best] += utility_score
                self.V[best] = value  # refresh with latest observed value
                return
        if len(self.K) >= self.capacity:
            worst = int(np.argmin(self.strength))
            self.K[worst], self.V[worst], self.strength[worst] = key, value, utility_score
        else:
            self.K = np.vstack([self.K, key[None, :]])
            self.V = np.vstack([self.V, value[None, :]])
            self.strength = np.append(self.strength, utility_score)

    def pop_promotable(self):
        """Return (key, value) pairs whose accumulated strength clears the
        promotion threshold, and remove them from the trace (section 9:
        p_promote = sigma(f_phi(T_t, S_t, U_hat)) -- approximated here by
        thresholding accumulated evidence directly)."""
        if len(self.K) == 0:
            return []
        ready = self.strength >= self.promote_threshold
        out = list(zip(self.K[ready], self.V[ready]))
        keep = ~ready
        self.K, self.V, self.strength = self.K[keep], self.V[keep], self.strength[keep]
        return out


def PersistentMemoryLikeCosSim(key, K):
    kn = key / (np.linalg.norm(key) + 1e-8)
    Kn = K / (np.linalg.norm(K, axis=-1, keepdims=True) + 1e-8)
    return Kn @ kn
