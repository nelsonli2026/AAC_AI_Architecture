"""
Persistent associative memory P_t (AAC spec sections 10-17, 21), using the
single memory-item schema (k_i, v_i, c_i, t_i, E_i) from sections 4/10 --
E_i (evidence/provenance) is implemented as a small integer "source count"
plus, for the credit-assignment pass, a running causal-utility estimate.

Implements:
  - sparse write                              (section 8)
  - associative read via softmax attention    (section 12)
  - optional k-means "hierarchical" routing   (section 11)
  - confidence reinforcement / decay          (section 15)
  - eviction below a confidence floor         (section 15)
  - compatible-merge vs. competing-hypothesis  (section 16)
  - ablation-based causal credit assignment,  (section 21, SIGN CORRECTED --
    C_i = ablated_loss - base_loss, positive when the memory helps, see
    AAC-corrections.md item 2)
"""
import numpy as np


class PersistentMemory:
    def __init__(self, d_key, d_val, max_items=512, evict_threshold=0.05,
                 merge_cos_threshold=0.97, n_clusters=8, seed=0):
        self.d_key, self.d_val = d_key, d_val
        self.max_items = max_items
        self.evict_threshold = evict_threshold
        self.merge_cos_threshold = merge_cos_threshold
        self.rng = np.random.default_rng(seed)

        self.K = np.zeros((0, d_key))      # keys
        self.V = np.zeros((0, d_val))      # values
        self.c = np.zeros((0,))            # confidence c_i
        self.t = np.zeros((0,), dtype=int)  # write time t_i
        self.evidence = np.zeros((0,), dtype=int)  # E_i: supporting-write count
        self.util_ema = np.zeros((0,))     # running causal-utility estimate

        # --- stats, purely for reporting ---
        self.n_writes = 0
        self.n_promotions = 0
        self.n_evictions = 0
        self.n_merges = 0

        # --- hierarchical routing (section 11) ---
        self.n_clusters = n_clusters
        self.centroids = None  # (n_clusters, d_key), rebuilt periodically

    # ------------------------------------------------------------------
    # write / promotion (funnel is: caller writes candidate key/value that
    # has already survived the temporary trace's promotion check)
    # ------------------------------------------------------------------
    def write(self, key, value, t_now, initial_conf=1.0):
        """Add one new item, or merge into a compatible existing one."""
        self.n_writes += 1
        if len(self.K) > 0:
            sims = self._cos_sim(key[None, :], self.K)[0]
            best = int(np.argmax(sims))
            if sims[best] >= self.merge_cos_threshold:
                same_value = np.allclose(self.V[best], value, atol=1e-3)
                if same_value:
                    # compatible memories: confidence-weighted merge (sec 16)
                    c_i, c_j = self.c[best], initial_conf
                    self.V[best] = (c_i * self.V[best] + c_j * value) / (c_i + c_j)
                    self.K[best] = (c_i * self.K[best] + c_j * key) / (c_i + c_j)
                    self.c[best] = c_i + c_j
                    self.evidence[best] += 1
                    self.n_merges += 1
                    return best
                # else: same key region, different value -> competing
                # hypothesis. Fall through and add as a *separate* item
                # rather than overwrite (section 14/16).
        self.K = np.vstack([self.K, key[None, :]])
        self.V = np.vstack([self.V, value[None, :]])
        self.c = np.append(self.c, initial_conf)
        self.t = np.append(self.t, t_now)
        self.evidence = np.append(self.evidence, 1)
        self.util_ema = np.append(self.util_ema, 0.0)
        self.n_promotions += 1
        return len(self.K) - 1

    # ------------------------------------------------------------------
    # hierarchical routing (section 11): cluster keys, route query to the
    # nearest cluster(s) first, then do detailed retrieval only within it.
    # ------------------------------------------------------------------
    def rebuild_routing(self, n_clusters=None):
        """n_clusters: override self.n_clusters for this rebuild (e.g. to
        test scaling the cluster count with sqrt(M) instead of a fixed
        constant -- see scaling_sweep.py)."""
        k = n_clusters if n_clusters is not None else self.n_clusters
        M = len(self.K)
        if M < k * 2:
            self.centroids = None
            return
        # a few iterations of lightweight k-means
        idx = self.rng.choice(M, k, replace=False)
        centroids = self.K[idx].copy()
        for _ in range(5):
            sims = self._cos_sim(self.K, centroids)
            assign = sims.argmax(axis=1)
            for c in range(k):
                members = self.K[assign == c]
                if len(members) > 0:
                    centroids[c] = members.mean(axis=0)
        self.centroids = centroids
        self._assign = self._cos_sim(self.K, centroids).argmax(axis=1)

    def _candidate_indices(self, query, top_clusters=2):
        """Return indices to search, using routing if available (O(d log M)
        cluster selection + O(Kd) local search) or all items otherwise."""
        M = len(self.K)
        if self.centroids is None or M == 0:
            return np.arange(M)
        csims = self._cos_sim(query[None, :], self.centroids)[0]
        chosen = np.argsort(-csims)[:top_clusters]
        mask = np.isin(self._assign, chosen)
        return np.where(mask)[0]

    # ------------------------------------------------------------------
    # associative read (section 12) with optional multi-query fan-out
    # (section 13) -- caller decides n_queries based on uncertainty.
    # ------------------------------------------------------------------
    def read(self, queries, top_k=4, use_routing=True):
        """queries: (n_q, d_key) -> (d_val,) weighted sum, plus indices used
        (for reinforcement / credit assignment bookkeeping)."""
        if len(self.K) == 0:
            return np.zeros(self.d_val), []
        used = set()
        m = np.zeros(self.d_val)
        for q in queries:
            cand = self._candidate_indices(q) if use_routing else np.arange(len(self.K))
            if len(cand) == 0:
                continue
            sims = self._cos_sim(q[None, :], self.K[cand])[0]
            k = min(top_k, len(cand))
            top = cand[np.argsort(-sims)[:k]]
            local_sims = self._cos_sim(q[None, :], self.K[top])[0]
            w = _softmax(local_sims)
            m += (w[:, None] * self.V[top]).sum(axis=0)
            used.update(top.tolist())
        return m / max(1, len(queries)), sorted(used)

    # ------------------------------------------------------------------
    # confidence reinforcement / decay (section 15)
    # ------------------------------------------------------------------
    def reinforce(self, indices, retrieval_evidence=1.0, utility=0.0,
                  eta_r=0.05, eta_u=0.1, eta_d=0.01):
        for i in indices:
            self.c[i] += eta_r * retrieval_evidence + eta_u * utility - eta_d
            self.evidence[i] += 1

    def decay_all(self, eta_d=0.002):
        self.c -= eta_d  # background forgetting pressure on unused items

    def evict(self):
        keep = self.c >= self.evict_threshold
        n_ev = int((~keep).sum())
        if n_ev:
            self.K, self.V = self.K[keep], self.V[keep]
            self.c, self.t = self.c[keep], self.t[keep]
            self.evidence, self.util_ema = self.evidence[keep], self.util_ema[keep]
            self.n_evictions += n_ev
            self.centroids = None  # routing structure is now stale
        # hard cap: if still over budget, drop the lowest-confidence items
        if len(self.K) > self.max_items:
            order = np.argsort(-self.c)[: self.max_items]
            self.K, self.V = self.K[order], self.V[order]
            self.c, self.t = self.c[order], self.t[order]
            self.evidence, self.util_ema = self.evidence[order], self.util_ema[order]

    # ------------------------------------------------------------------
    # ablation-based causal credit assignment (section 21, sign corrected)
    #   C_i = L_ablated(i) - L_base      (positive => memory i helps)
    #   U_i <- U_i + eta * C_i
    # eval_fn(mask) must return a scalar loss when memory items where
    # mask==False are hidden from retrieval (mask is a boolean array over
    # the *current* memory items).
    # ------------------------------------------------------------------
    def ablation_credit_assignment(self, eval_fn, sample_indices, eta=0.2):
        if len(self.K) == 0:
            return
        full_mask = np.ones(len(self.K), dtype=bool)
        base_loss = eval_fn(full_mask)
        for i in sample_indices:
            mask = full_mask.copy()
            mask[i] = False
            ablated_loss = eval_fn(mask)
            C_i = ablated_loss - base_loss  # corrected sign
            self.util_ema[i] = 0.9 * self.util_ema[i] + 0.1 * C_i
            self.c[i] += eta * C_i

    @staticmethod
    def _cos_sim(A, B):
        An = A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-8)
        Bn = B / (np.linalg.norm(B, axis=-1, keepdims=True) + 1e-8)
        return An @ Bn.T

    def __len__(self):
        return len(self.K)


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
