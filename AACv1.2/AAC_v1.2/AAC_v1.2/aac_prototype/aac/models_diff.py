import numpy as np
from aac import autodiff as ad


class AdamTensor:
    def __init__(self, params, lr=5e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.t = 0

    def step(self):
        self.t += 1
        for k, p in self.params.items():
            g = p.grad
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def zero_grad(self):
        for p in self.params.values():
            p.zero_grad()


def _init(rng, shape, fan_in=None):
    fan_in = fan_in or (shape[-1] if len(shape) > 1 else shape[0])
    return ad.Tensor(rng.normal(0, 1.0 / np.sqrt(fan_in), size=shape),
                      requires_grad=True)


def _zeros(shape):
    return ad.Tensor(np.zeros(shape), requires_grad=True)


class RNNBaseline:
    name = "RNN-only (no memory)"

    def __init__(self, vocab_size, d_emb=32, d_h=32, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "Emb": _init(rng, (vocab_size, d_emb), fan_in=d_emb),
            "Whh": _init(rng, (d_h, d_h)),
            "Wxh": _init(rng, (d_h, d_emb)),
            "bh": _zeros((d_h,)),
            "Wout": _init(rng, (vocab_size, d_h)),
            "bout": _zeros((vocab_size,)),
        }
        self.opt = AdamTensor(self.params)
        self.d_h = d_h

    def forward_episode(self, episode, train=True):
        p = self.params
        tokens = episode["tokens"]
        qpos = dict(zip(episode["query_positions"], episode["query_labels"]))
        h = ad.Tensor(np.zeros(self.d_h))
        losses, correct, total = [], 0, 0

        for t, tok in enumerate(tokens):
            emb = ad.embedding_lookup(p["Emb"], int(tok))
            h = ad.tanh(ad.add(ad.add(ad.matvec(p["Whh"], h),
                                       ad.matvec(p["Wxh"], emb)), p["bh"]))
            if t in qpos:
                logits = ad.add(ad.matvec(p["Wout"], h), p["bout"])
                label = qpos[t]
                loss = ad.softmax_cross_entropy(logits, label)
                losses.append(loss)
                pred = int(np.argmax(logits.data))
                correct += int(pred == label)
                total += 1

        return self._finish(losses, correct, total, train)

    def _finish(self, losses, correct, total, train):
        if not losses:
            return 0.0, 0.0
        total_loss = losses[0]
        for l in losses[1:]:
            total_loss = ad.add(total_loss, l)
        if train:
            total_loss.backward()
        else:
            # eval mode never calls .backward(), so the cycle-breaking
            # that lives there never runs -- do it explicitly here instead,
            # or every eval call leaks its whole graph (see autodiff.py's
            # free_graph docstring for the full explanation).
            ad.free_graph(total_loss)
        return float(total_loss.data) / total, correct / total


class AttentionBaseline:
    """One-layer causal self-attention over the full token history. This
    model's memory is UNCOMPRESSED -- it never forgets or throws anything
    away, so its capacity grows with sequence length. It's a ceiling on
    achievable accuracy, not a fair capacity match against AACDiffModel's
    fixed-size memory (see PHASE2_RESULTS.md)."""
    name = "Single-layer causal self-attention (uncompressed KV)"

    def __init__(self, vocab_size, d_emb=32, d_h=32, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "Emb": _init(rng, (vocab_size, d_emb), fan_in=d_emb),
            "Wq": _init(rng, (d_h, d_emb)),
            "Wk": _init(rng, (d_h, d_emb)),
            "Wv": _init(rng, (d_h, d_emb)),
            "Wcomb": _init(rng, (d_h, d_emb + d_h)),
            "bcomb": _zeros((d_h,)),
            "Wout": _init(rng, (vocab_size, d_h)),
            "bout": _zeros((vocab_size,)),
        }
        self.opt = AdamTensor(self.params)
        self.d_h = d_h

    def forward_episode(self, episode, train=True):
        p = self.params
        tokens = episode["tokens"]
        qpos = dict(zip(episode["query_positions"], episode["query_labels"]))
        K_hist, V_hist = [], []
        losses, correct, total = [], 0, 0
        scale = 1.0 / np.sqrt(self.d_h)
        prev_emb = None

        for t, tok in enumerate(tokens):
            emb = ad.embedding_lookup(p["Emb"], int(tok))
            if prev_emb is not None:
                # Bind (previous token as key) -> (current token as value):
                # this is what makes "attend back to an earlier occurrence
                # of this token" actually recover the token that FOLLOWED
                # it, rather than the key's own embedding again (see the
                # induction-head binding-shift bug fix in PHASE2_RESULTS.md).
                k = ad.matvec(p["Wk"], prev_emb)
                v = ad.matvec(p["Wv"], emb)
                K_hist.append(k)
                V_hist.append(v)

            if t in qpos and K_hist:
                q = ad.matvec(p["Wq"], emb)
                K_stack = ad.stack(K_hist)
                V_stack = ad.stack(V_hist)
                scores = ad.mul_const(ad.matvec(K_stack, q), scale)
                weights = ad.softmax(scores)
                read = ad.vecmat(weights, V_stack)
                comb = ad.concat(emb, read)
                hid = ad.tanh(ad.add(ad.matvec(p["Wcomb"], comb), p["bcomb"]))
                logits = ad.add(ad.matvec(p["Wout"], hid), p["bout"])
                label = qpos[t]
                loss = ad.softmax_cross_entropy(logits, label)
                losses.append(loss)
                pred = int(np.argmax(logits.data))
                correct += int(pred == label)
                total += 1

            prev_emb = emb

        return RNNBaseline._finish(self, losses, correct, total, train)


class AACDiffModel:
    name = "AAC-diff (fast state + gated fixed-size associative memory)"

    def __init__(self, vocab_size, d_emb=32, d_h=32, decay=0.95, beta=0.1,
                 update_rule="hebbian", gate_mode="learned", gate_bias_init=0.0,
                 aux_gate_weight=0.0, aux_pos_weight=1.0, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "Emb": _init(rng, (vocab_size, d_emb), fan_in=d_emb),
            "Whh": _init(rng, (d_h, d_h)),
            "Wxh": _init(rng, (d_h, d_emb)),
            "bh": _zeros((d_h,)),
            "Wk": _init(rng, (d_h, d_emb)),
            "Wv": _init(rng, (d_h, d_emb)),
            "Wq": _init(rng, (d_h, d_emb)),
            "Wg": _init(rng, (1, d_h)),
            # gate bias: >0 warm-starts the gate toward "open" (sigmoid(bg))
            "bg": ad.Tensor(np.array([gate_bias_init]), requires_grad=True),
            "Wcomb": _init(rng, (d_h, 2 * d_h)),
            "bcomb": _zeros((d_h,)),
            "Wout": _init(rng, (vocab_size, d_h)),
            "bout": _zeros((vocab_size,)),
        }
        self.opt = AdamTensor(self.params)
        self.d_h = d_h
        self.decay = decay
        self.beta = beta
        self.update_rule = update_rule
        self.gate_mode = gate_mode  # "learned", "oracle", "oracle_decay", "curriculum"
        self.alpha = 1.0  # curriculum mixing weight: 0 = pure oracle write,
        # 1 = pure learned-gate write. Gate is ALWAYS computed & trained via
        # backprop in curriculum mode; alpha only controls how much of the
        # actual M update comes from the learned gate vs. the oracle mask.
        # Auxiliary supervised loss on the gate itself: BCE(g_t, oracle_label_t)
        # added to the main task loss, weighted by aux_gate_weight (0 = off,
        # i.e. exactly the old unsupervised "learned" mode). aux_pos_weight
        # upweights the rare positive (key-write) class against the far more
        # common filler class.
        self.aux_gate_weight = aux_gate_weight
        self.aux_pos_weight = aux_pos_weight
        self.last_gate_values = []
        self.grad_norms = {"Wg": [], "Wk": [], "Wv": [], "Wq": []}

    def forward_episode(self, episode, train=True):
        p = self.params
        tokens = episode["tokens"]
        qpos = dict(zip(episode["query_positions"], episode["query_labels"]))
        oracle_positions = set(int(pos) for pos in
                                [kp + 1 for kp in episode["key_positions"] if kp + 1 < len(tokens)])
        h = ad.Tensor(np.zeros(self.d_h))
        M = ad.Tensor(np.zeros((self.d_h, self.d_h)))
        losses, correct, total = [], 0, 0
        aux_losses = []
        self.last_gate_values = []
        prev_emb = None

        for t, tok in enumerate(tokens):
            emb = ad.embedding_lookup(p["Emb"], int(tok))
            h = ad.tanh(ad.add(ad.add(ad.matvec(p["Whh"], h),
                                       ad.matvec(p["Wxh"], emb)), p["bh"]))

            if prev_emb is not None:
                k = ad.matvec(p["Wk"], prev_emb)
                v = ad.matvec(p["Wv"], emb)
                if self.gate_mode == "oracle":
                    # oracle: hard 1/0 write at exactly the correct positions,
                    # NOT a function of any trainable param -> no gradient
                    # flows to a "gate", by construction. Wg/bg are unused.
                    # NOTE: no decay applied here (accumulates undamped).
                    if t in oracle_positions:
                        M = ad.add(M, ad.outer(k, v))
                    self.last_gate_values.append(1.0 if t in oracle_positions else 0.0)

                elif self.gate_mode == "oracle_decay":
                    # controlled variant: SAME hard oracle write timing as
                    # "oracle", but WITH the same decay=0.95 applied every
                    # step as the learned/curriculum paths use. Isolates
                    # "correct write timing" from "no decay at all".
                    oracle_val = 1.0 if t in oracle_positions else 0.0
                    M = ad.add(ad.mul_const(M, self.decay),
                               ad.mul_const(ad.outer(k, v), oracle_val))
                    self.last_gate_values.append(oracle_val)

                elif self.gate_mode == "curriculum":
                    # gate is ALWAYS computed and trained (real gradient to
                    # Wg/bg every step), but the write actually applied to M
                    # is a blend of the learned gate and the oracle mask.
                    # alpha=0 -> pure oracle write (like oracle mode) but
                    # gate is still being trained on the side; alpha=1 ->
                    # pure learned-gate write (like learned mode). Ramp
                    # alpha 0->1 across training (set externally per-episode).
                    g = ad.sigmoid(ad.add(ad.matvec(p["Wg"], h), p["bg"]))
                    self.last_gate_values.append(float(g.data[0]))
                    oracle_val = 1.0 if t in oracle_positions else 0.0
                    # g_eff = alpha*g + (1-alpha)*oracle_val  (oracle_val is a
                    # python float constant, not a Tensor -> no extra graph)
                    g_eff_data = self.alpha * g.data + (1 - self.alpha) * oracle_val
                    g_eff = ad.Tensor(g_eff_data, _children=(g,), _op="curric_blend")

                    def _mk_backward(gate_t, alpha_local, out_t):
                        # bind gate_t/alpha_local/out_t as defaults so each
                        # timestep's closure captures ITS OWN tensors, not
                        # whatever the loop variables point to by the time
                        # .backward() actually runs (late-binding bug fixed).
                        def _b():
                            if gate_t.requires_grad:
                                gate_t.grad += out_t.grad * alpha_local
                        return _b
                    g_eff._backward = _mk_backward(g, self.alpha, g_eff)

                    M = ad.add(ad.mul_const(M, self.decay), ad.scale(ad.outer(k, v), g_eff))

                else:  # "learned" (with optional warm-start bias bg)
                    g = ad.sigmoid(ad.add(ad.matvec(p["Wg"], h), p["bg"]))
                    self.last_gate_values.append(float(g.data[0]))
                    if self.aux_gate_weight > 0.0:
                        oracle_val = 1.0 if t in oracle_positions else 0.0
                        aux_losses.append(ad.mul_const(
                            ad.bce_loss(g, oracle_val, pos_weight=self.aux_pos_weight),
                            self.aux_gate_weight))
                    M = ad.add(ad.mul_const(M, self.decay), ad.scale(ad.outer(k, v), g))
            else:
                self.last_gate_values.append(0.0)

            if t in qpos:
                q = ad.matvec(p["Wq"], emb)
                read = ad.vecmat(q, M)
                comb = ad.concat(h, read)
                hid = ad.tanh(ad.add(ad.matvec(p["Wcomb"], comb), p["bcomb"]))
                logits = ad.add(ad.matvec(p["Wout"], hid), p["bout"])
                label = qpos[t]
                loss = ad.softmax_cross_entropy(logits, label)
                losses.append(loss)
                pred = int(np.argmax(logits.data))
                correct += int(pred == label)
                total += 1

            prev_emb = emb

        result = self._finish_with_aux(losses, aux_losses, correct, total, train)
        if train and self.opt.t <= 500 and self.opt.t % 50 == 0:
            for key in ["Wg", "Wk", "Wv", "Wq"]:
                if key in p:
                    gnorm = float(np.linalg.norm(p[key].grad))
                    self.grad_norms[key].append((self.opt.t, gnorm))
        return result

    def _finish_with_aux(self, losses, aux_losses, correct, total, train):
        """Same as RNNBaseline._finish, but adds the auxiliary gate-BCE
        terms into the SAME backward pass (so Wg/bg get gradient from both
        the task loss and the direct supervision), while still reporting
        only the task loss/accuracy for comparability with the unsupervised
        runs."""
        if not losses:
            return 0.0, 0.0
        total_loss = losses[0]
        for l in losses[1:]:
            total_loss = ad.add(total_loss, l)
        task_loss_val = float(total_loss.data)
        combined = total_loss
        for al in aux_losses:
            combined = ad.add(combined, al)
        if train:
            combined.backward()
        else:
            # same fix as RNNBaseline._finish -- eval-mode never calls
            # .backward(), so break the cycles explicitly here instead.
            ad.free_graph(combined)
        return task_loss_val / total, correct / total
