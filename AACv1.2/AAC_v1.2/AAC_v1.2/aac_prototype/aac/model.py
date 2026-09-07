"""
Full AAC forward pass over a sequence, following the corrected (non-
circular) update order from AAC-corrections.md item 1:

    e_t = E(x_t, S_{t-1})
    q_t = Q(e_t, S_{t-1}, T_{t-1})
    a_t = pi(S_{t-1}, T_{t-1}, P_{t-1})
    m_t = Read(P_{t-1}, q_t)
    S_t = F_S(S_{t-1}, x_t, a_t)
    T_t = lambda_t T_{t-1} + F_T(x_t, S_t)
    h_t = F_H(x_t, S_t, T_t, m_t)
    dP_t = Validate(T_t, a_t)
    P_t = Merge(Evict(P_{t-1} + dP_t))

Every quantity is computed strictly from already-resolved values -- no
symbol is used before it is defined for this timestep.
"""
import numpy as np
from .fast_state import FastState
from .memory import PersistentMemory
from .trace import TemporaryTrace
from .controller import Controller
from .mlp import MLP
from task import decode_token, KEY, VAL, QUERY, FILLER


def _entropy(p):
    p = np.clip(p, 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())


class AACModel:
    def __init__(self, n_ids, n_filler_vocab, d_id=16, d_emb=24, d_state=24,
                 seed=0):
        self.n_ids = n_ids
        vocab_size = 3 * n_ids + n_filler_vocab
        rng = np.random.default_rng(seed)

        # Fixed "already-good" token representations (see model design note
        # in README): identity vectors shared between KEY_i and QUERY_i so
        # the memory can actually match a later query to an earlier key.
        self.Id = rng.normal(0, 1.0, size=(n_ids, d_id))
        self.ValEmb = rng.normal(0, 1.0, size=(n_ids, d_id))
        self.Emb = rng.normal(0, 1.0 / np.sqrt(d_emb), size=(vocab_size, d_emb))

        self.fast_state = FastState(d_state, d_emb, rank=6, seed=seed)
        self.controller = Controller(d_in=4 + d_id + d_state, d_hidden=32,
                                      seed=seed)
        self.readout = MLP(d_state + d_id, 32, n_ids, head="softmax", seed=seed)

        self.d_id, self.d_state = d_id, d_state

    # ------------------------------------------------------------------
    def run_episode(self, episode, train=True, use_routing=True,
                     collect_ablation=False):
        tokens, kinds = episode["tokens"], episode["kinds"]
        qpos, qlab = episode["query_positions"], episode["query_labels"]
        qpos_set = dict(zip(qpos, qlab))

        memory = PersistentMemory(self.d_id, self.d_id, seed=0)
        trace = TemporaryTrace(self.d_id, self.d_id, seed=0)
        S_prev = self.fast_state.init_state(1)[0]
        pending_key = None

        correct, total = 0, 0
        query_log = []  # (t, query_ident, true_label) for optional ablation study

        for t, tok in enumerate(tokens):
            kind, ident = decode_token(int(tok), self.n_ids)
            type_onehot = np.eye(4)[kind]

            if kind == KEY:
                cand_vec = self.Id[ident]
            elif kind == VAL:
                cand_vec = self.ValEmb[ident]
            elif kind == QUERY:
                cand_vec = self.Id[ident]
            else:
                cand_vec = np.zeros(self.d_id)

            # ---- e_t, controller utility estimate (uses S_{t-1}) ----
            feat = np.concatenate([type_onehot, cand_vec, S_prev])
            u_hat = self.controller.estimate_utility(feat[None, :])[0]
            proxy_label = 1.0 if kind in (KEY, VAL) else 0.0
            if train:
                self.controller.train_utility(feat[None, :],
                                               np.array([proxy_label]))

            # ---- a_t: write gating funnels KEY/VAL evidence into T_t ----
            if kind == KEY:
                pending_key = self.Id[ident]
            elif kind == VAL and pending_key is not None:
                if self.controller.gate_write(u_hat):
                    trace.add_candidate(pending_key, cand_vec, u_hat)
                pending_key = None

            trace.step_decay()
            for k, v in trace.pop_promotable():
                memory.write(k, v, t_now=t)
            if t % 25 == 0:
                memory.decay_all()
                memory.evict()
                memory.rebuild_routing()

            # ---- q_t / m_t: read (only meaningful at QUERY tokens) ----
            if kind == QUERY:
                self.controller.n_read_attempts += 1
                val, used_idx = memory.read([cand_vec], top_k=4,
                                             use_routing=use_routing)
                readout_in = np.concatenate([S_prev, val])
                logits = self.readout.forward(readout_in[None, :],
                                               train=train)[0]
                ent = _entropy(logits)
                extra = self.controller.reasoning_budget(ent)
                for _ in range(extra):
                    nq = self.controller.n_queries(ent)
                    val, used_idx = memory.read([cand_vec] * nq, top_k=4,
                                                 use_routing=use_routing)
                    readout_in = np.concatenate([S_prev, val])
                    logits = self.readout.forward(readout_in[None, :],
                                                   train=train)[0]
                    ent = _entropy(logits)

                true_label = qpos_set[t]
                pred = int(np.argmax(logits))
                correct += int(pred == true_label)
                total += 1
                if train:
                    self.readout.backward_softmax_xent(np.array([true_label]))
                prob_correct = float(logits[true_label])
                memory.reinforce(used_idx, retrieval_evidence=1.0,
                                  utility=prob_correct)
                if collect_ablation:
                    query_log.append((t, cand_vec.copy(), true_label))

            # ---- S_t: advance fast state last, using only past values ----
            emb = self.Emb[int(tok)]
            S_t = self.fast_state.step(S_prev[None, :], emb[None, :])[0]
            S_prev = S_t

        memory.decay_all()
        memory.evict()

        stats = dict(accuracy=correct / max(1, total),
                     n_writes=memory.n_writes,
                     n_promotions=memory.n_promotions,
                     n_evictions=memory.n_evictions,
                     n_merges=memory.n_merges,
                     final_memory_size=len(memory),
                     seq_len=len(tokens))
        return stats, memory, query_log, S_prev

    # ------------------------------------------------------------------
    def ablation_study(self, memory, query_log, S_prev, n_items_to_test=5):
        """Demonstrate section 21 (sign-corrected): re-evaluate this
        episode's queries with individual memory items hidden, and update
        their confidence via C_i = L_ablated(i) - L_base."""
        if len(memory) == 0 or not query_log:
            return None

        def eval_fn(mask):
            losses = []
            for (t, q, label) in query_log:
                if mask.sum() == 0:
                    m = np.zeros(self.d_id)
                else:
                    sims = memory._cos_sim(q[None, :], memory.K[mask])[0]
                    w_idx = np.argsort(-sims)[: min(4, mask.sum())]
                    local = memory._cos_sim(q[None, :], memory.K[mask][w_idx])[0]
                    from .memory import _softmax
                    w = _softmax(local)
                    m = (w[:, None] * memory.V[mask][w_idx]).sum(axis=0)
                logits = self.readout.forward(
                    np.concatenate([S_prev, m])[None, :])[0]
                p = np.clip(logits[label], 1e-9, 1.0)
                losses.append(-np.log(p))
            return float(np.mean(losses))

        rng = np.random.default_rng(0)
        sample = rng.choice(len(memory), size=min(n_items_to_test, len(memory)),
                             replace=False)
        before = memory.c[sample].copy()
        memory.ablation_credit_assignment(eval_fn, sample)
        after = memory.c[sample]
        return list(zip(sample.tolist(), before.tolist(), after.tolist()))
