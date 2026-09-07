"""
The real test: train RNNBaseline, AttentionBaseline, and AACDiffModel
end-to-end (full BPTT via aac/autodiff.py) on the harder MQAR task
(task_mqar.py -- shared vocabulary, no privileged key/query alignment),
with matched embedding/hidden dims and identical optimizer settings.

Reports:
  - training accuracy curve for all three
  - held-out accuracy
  - accuracy vs. noise-gap length (does AAC-diff's compressed memory
    degrade differently than the attention baseline's uncompressed one?)
  - a diagnostic on AAC-diff's learned write gate: did it learn, purely
    from the recall loss (no explicit supervision), to write more
    strongly at key positions than at filler positions?
  - update-rule comparison (Hebbian / regularized delta-rule / Widrow-Hoff)
  - a write-oracle ablation, and a perfect-write read-harness check

NOTE (Phase 3): the gate diagnostics, perfect-write test, and "dual
bottleneck" framing below are the ORIGINAL Phase 2 / delta-rule-phase
versions, preserved for provenance. PHASE3_GATE_RESULTS.md substantially
revises several of this file's conclusions -- most importantly, the
perfect-write test's "read mechanism might be independently broken"
reading does not survive joint training from scratch with an oracle gate
(see PHASE3_GATE_RESULTS.md, Finding 1), and Phase 3's own headline
findings were themselves later found to be dominated by an unscheduled
learning rate, not gate selectivity (see PHASE3_GATE_RESULTS.md,
"Isolating the LR-decay effect from gate supervision"). Read this file's
numbers in that light, not as a final word.
"""
import sys, time
import numpy as np

sys.path.insert(0, ".")
from task_mqar import MQARTask
from aac.models_diff import RNNBaseline, AttentionBaseline, AACDiffModel, AdamTensor

D_EMB, D_H = 32, 32
N_VOCAB = 64
SEED = 0
LR = 1e-2
BATCH = 8


def train_model(model_cls, task, n_episodes, eval_every=250, seed=SEED,
                 lr=LR, batch=BATCH):
    model = model_cls(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=seed)
    model.opt = AdamTensor(model.params, lr=lr)
    accs = []
    t0 = time.perf_counter()
    model.opt.zero_grad()
    for ep in range(1, n_episodes + 1):
        episode = task.sample_episode(n_pairs=6, n_queries=6)
        loss, acc = model.forward_episode(episode, train=True)
        accs.append(acc)
        if ep % batch == 0:
            model.opt.step()
            model.opt.zero_grad()
        if ep % eval_every == 0:
            dt = time.perf_counter() - t0
            print(f"  [{model.name:55s}] ep {ep:5d}  "
                  f"train_acc(last {eval_every})={np.mean(accs[-eval_every:]):.3f}  "
                  f"({dt:.1f}s elapsed)")
    return model, accs


def evaluate(model, task, n_episodes=150, gap_len=(20, 40), seed=999):
    eval_task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    accs = []
    for _ in range(n_episodes):
        episode = eval_task.sample_episode(n_pairs=6, n_queries=6, gap_len=gap_len)
        _, acc = model.forward_episode(episode, train=False)
        accs.append(acc)
    return float(np.mean(accs))


def gate_diagnostic(model, task, n_episodes=500, seed=777):
    """Does AAC-diff's learned write gate fire more strongly at key
    positions than at filler positions -- purely emergent from the
    recall loss, with no supervision telling it which is which?

    CORRECTED (Phase 2 -> delta-rule phase): the memory write at timestep
    t binds (token[t-1], token[t]), so the key-value write that matters is
    at gates[key_position + 1], not gates[key_position] (the original
    off-by-one; see DELTA_RULE_RESULTS.md)."""
    eval_task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    key_gates, filler_gates = [], []
    for _ in range(n_episodes):
        episode = eval_task.sample_episode(n_pairs=6, n_queries=6)
        model.forward_episode(episode, train=False)
        gates = model.last_gate_values
        key_positions = episode["key_positions"]
        key_write_indices = set()
        for kp in key_positions:
            if kp + 1 < len(gates):
                key_write_indices.add(kp + 1)
        for t, g in enumerate(gates):
            (key_gates if t in key_write_indices else filler_gates).append(g)
    return float(np.mean(key_gates)), float(np.mean(filler_gates))


def perfect_write_test(model, task, n_episodes=500, seed=888):
    """Isolate read-side from write-side: construct M perfectly by directly
    writing the true (key, value) pairs from each episode, then test if the
    trained query projection and readout head can correctly retrieve values.

    CAUTION (superseded, see module docstring): this test freezes a
    perfect M on top of Wq/Wk/Wv/Wcomb/Wout weights that were trained
    under the normal (possibly broken) regime. Near-chance accuracy here
    does NOT establish that the read mechanism is independently broken --
    see PHASE3_GATE_RESULTS.md Finding 1 for the joint-training result
    that overturns this reading."""
    from aac import autodiff as ad
    eval_task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    correct, total = 0, 0

    for _ in range(n_episodes):
        episode = eval_task.sample_episode(n_pairs=6, n_queries=6)
        tokens = episode["tokens"]
        qpos = dict(zip(episode["query_positions"], episode["query_labels"]))
        key_positions = episode["key_positions"]

        M = ad.Tensor(np.zeros((model.d_h, model.d_h)))
        p = model.params
        for kp in key_positions:
            if kp + 1 < len(tokens):
                key_tok = int(tokens[kp])
                val_tok = int(tokens[kp + 1])
                key_emb = ad.embedding_lookup(p["Emb"], key_tok)
                val_emb = ad.embedding_lookup(p["Emb"], val_tok)
                k = ad.matvec(p["Wk"], key_emb)
                v = ad.matvec(p["Wv"], val_emb)
                M = ad.add(M, ad.outer(k, v))  # perfect writes, gate=1.0

        h = ad.Tensor(np.zeros(model.d_h))  # dummy hidden state
        for q_pos, q_label in qpos.items():
            query_tok = int(tokens[q_pos])
            query_emb = ad.embedding_lookup(p["Emb"], query_tok)
            q = ad.matvec(p["Wq"], query_emb)
            read = ad.vecmat(q, M)
            comb = ad.concat(h, read)
            hid = ad.tanh(ad.add(ad.matvec(p["Wcomb"], comb), p["bcomb"]))
            logits = ad.add(ad.matvec(p["Wout"], hid), p["bout"])
            pred = int(np.argmax(logits.data))
            correct += int(pred == q_label)
            total += 1

    return float(correct / total) if total > 0 else 0.0


if __name__ == "__main__":
    task = MQARTask(n_vocab=N_VOCAB, seed=SEED)
    n_episodes = 3000

    results = {}

    update_rules = [
        ("hebbian", "AAC-diff (Hebbian)"),
    ]

    for cls in [RNNBaseline, AttentionBaseline]:
        print(f"\n=== Training {cls.name} ===")
        model, accs = train_model(cls, task, n_episodes)
        results[cls.__name__] = dict(model=model, accs=accs)

    for rule, rule_name in update_rules:
        print(f"\n=== Training {rule_name} ===")
        model = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=SEED,
                              update_rule=rule, gate_mode="learned")
        model.opt = AdamTensor(model.params, lr=LR)
        accs = []
        t0 = time.perf_counter()
        model.opt.zero_grad()
        for ep in range(1, n_episodes + 1):
            episode = task.sample_episode(n_pairs=6, n_queries=6)
            loss, acc = model.forward_episode(episode, train=True)
            accs.append(acc)
            if ep % BATCH == 0:
                model.opt.step()
                model.opt.zero_grad()
            if ep % 250 == 0:
                dt = time.perf_counter() - t0
                print(f"  [{rule_name:55s}] ep {ep:5d}  "
                      f"train_acc(last 250)={np.mean(accs[-250:]):.3f}  "
                      f"({dt:.1f}s elapsed)")
        results[f"AACDiffModel_{rule}"] = dict(model=model, accs=accs, rule=rule)

    print(f"\n=== Training AAC-diff write-oracle ablation ===")
    oracle_model = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H,
                                 seed=SEED, update_rule="hebbian", gate_mode="oracle")
    oracle_model.opt = AdamTensor(oracle_model.params, lr=LR)
    oracle_accs = []
    t0 = time.perf_counter()
    oracle_model.opt.zero_grad()
    for ep in range(1, n_episodes + 1):
        episode = task.sample_episode(n_pairs=6, n_queries=6)
        loss, acc = oracle_model.forward_episode(episode, train=True)
        oracle_accs.append(acc)
        if ep % BATCH == 0:
            oracle_model.opt.step()
            oracle_model.opt.zero_grad()
        if ep % 250 == 0:
            dt = time.perf_counter() - t0
            print(f"  [{oracle_model.name:55s}] ep {ep:5d}  "
                  f"train_acc(last 250)={np.mean(oracle_accs[-250:]):.3f}  "
                  f"({dt:.1f}s elapsed)")
    results["AACDiffModel_write_oracle"] = dict(model=oracle_model, accs=oracle_accs, rule="oracle")

    print("\n=== Held-out evaluation (same distribution as training) ===")
    for name, r in results.items():
        acc = evaluate(r["model"], task)
        r["heldout_acc"] = acc
        model_name = r["model"].name if hasattr(r["model"], "name") else name
        print(f"  {model_name:55s}  acc={acc:.3f}")

    print("\n=== Generalization: accuracy vs. noise-gap length "
          "(never seen this long during training) ===")
    gaps = [(20, 40), (60, 90), (150, 200), (300, 400)]
    print(f"  {'model':55s}  " + "  ".join(f"gap~{g[0]}-{g[1]:<4d}" for g in gaps))
    for name, r in results.items():
        accs_by_gap = [evaluate(r["model"], task, n_episodes=80, gap_len=g)
                       for g in gaps]
        r["gap_curve"] = accs_by_gap
        model_name = r["model"].name if hasattr(r["model"], "name") else name
        print(f"  {model_name:55s}  " +
              "  ".join(f"{a:10.3f}" for a in accs_by_gap))

    print("\n=== AAC-diff gate diagnostic (key->value write at position+1) ===")
    for name, r in results.items():
        if "AACDiffModel" in name:
            key_g, filler_g = gate_diagnostic(r["model"], task, n_episodes=500)
            ratio = key_g / max(filler_g, 1e-6)
            model_name = r["model"].name if hasattr(r["model"], "name") else name
            print(f"  {model_name} ({r['rule']}): key={key_g:.3f}  filler={filler_g:.3f}  "
                  f"ratio={ratio:.2f}x  {'(selectivity!)' if ratio > 1.2 else '(no selectivity)'}")

    print("\n=== Perfect-write read harness check (see caution in module docstring) ===")
    for name, r in results.items():
        if "AACDiffModel" in name and r["rule"] != "oracle":
            acc = perfect_write_test(r["model"], task, n_episodes=500)
            model_name = r["model"].name if hasattr(r["model"], "name") else name
            print(f"  {model_name} ({r['rule']}): perfect-M accuracy = {acc:.3f}")

    print("\n=== Summary ===")
    for name, r in results.items():
        model_name = r["model"].name if hasattr(r["model"], "name") else name
        print(f"  {model_name:55s}  held-out={r['heldout_acc']:.3f}  "
              f"gap-300-400={r['gap_curve'][-1]:.3f}")
