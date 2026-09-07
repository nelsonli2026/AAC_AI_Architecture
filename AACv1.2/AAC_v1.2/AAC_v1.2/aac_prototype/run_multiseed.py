import sys, time
import numpy as np

sys.path.insert(0, ".")
from task_mqar import MQARTask
from aac.models_diff import AACDiffModel, AdamTensor

D_EMB, D_H = 32, 32
N_VOCAB = 64
LR = 1e-2
BATCH = 8
N_EPISODES = 2000


def evaluate(model, n_episodes=200, gap_len=(20, 40), seed=999):
    eval_task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    accs = []
    for _ in range(n_episodes):
        episode = eval_task.sample_episode(n_pairs=6, n_queries=6, gap_len=gap_len)
        _, acc = model.forward_episode(episode, train=False)
        accs.append(acc)
    return float(np.mean(accs))


def gate_diagnostic(model, n_episodes=300, seed=777):
    eval_task = MQARTask(n_vocab=N_VOCAB, seed=seed)
    key_gates, filler_gates = [], []
    for _ in range(n_episodes):
        episode = eval_task.sample_episode(n_pairs=6, n_queries=6)
        model.forward_episode(episode, train=False)
        gates = model.last_gate_values
        key_write_idx = set(kp + 1 for kp in episode["key_positions"] if kp + 1 < len(gates))
        for t, g in enumerate(gates):
            (key_gates if t in key_write_idx else filler_gates).append(g)
    return float(np.mean(key_gates)), float(np.mean(filler_gates))


def train_and_eval(seed, n_episodes=N_EPISODES):
    task = MQARTask(n_vocab=N_VOCAB, seed=seed)  # task sampling ALSO seeded per-seed,
    # so each run is a genuinely independent trial (different episodes AND different
    # weight init), not just a different weight init replaying identical episodes.
    model = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=seed,
                          gate_mode="learned", decay=0.999,
                          aux_gate_weight=1.0, aux_pos_weight=5.0)
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
        if ep % 500 == 0:
            print(f"    seed {seed}  ep {ep:5d}  train_acc(last500)={np.mean(accs[-500:]):.3f}  "
                  f"({time.perf_counter()-t0:.1f}s)")
    held_out = evaluate(model, seed=999)  # fixed eval seed across all runs, for comparability
    key_g, filler_g = gate_diagnostic(model, seed=777)
    ratio = key_g / max(filler_g, 1e-6)
    return dict(seed=seed, held_out=held_out, key_gate=key_g, filler_gate=filler_g, ratio=ratio,
                final_train_acc=float(np.mean(accs[-250:])))


if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    results = []
    for s in seeds:
        print(f"=== seed {s} ===")
        r = train_and_eval(s)
        results.append(r)
        print(f"  seed {s}: held_out={r['held_out']:.3f}  ratio={r['ratio']:.2f}x  "
              f"key={r['key_gate']:.3f}  filler={r['filler_gate']:.3f}")

    held_outs = np.array([r["held_out"] for r in results])
    ratios = np.array([r["ratio"] for r in results])

    print("\n=== Summary across seeds ===")
    print(f"{'seed':>6} {'held_out':>10} {'gate_ratio':>12} {'key_gate':>10} {'filler_gate':>12}")
    for r in results:
        print(f"{r['seed']:>6} {r['held_out']:>10.3f} {r['ratio']:>12.2f} {r['key_gate']:>10.3f} {r['filler_gate']:>12.3f}")

    print(f"\nheld-out accuracy: mean={held_outs.mean():.3f}  std={held_outs.std():.3f}  "
          f"min={held_outs.min():.3f}  max={held_outs.max():.3f}")
    print(f"gate ratio:        mean={ratios.mean():.2f}x  std={ratios.std():.2f}  "
          f"min={ratios.min():.2f}x  max={ratios.max():.2f}x")
