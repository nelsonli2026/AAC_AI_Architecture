import sys, time
import numpy as np

sys.path.insert(0, ".")
from task_mqar import MQARTask
from aac.models_diff import AACDiffModel, AdamTensor

D_EMB, D_H = 32, 32
N_VOCAB = 64
SEED = 0
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


def train(task, gate_mode, decay, n_episodes=N_EPISODES, label=""):
    model = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=SEED,
                          gate_mode=gate_mode, decay=decay)
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
            print(f"  [{label:34s}] ep {ep:5d}  train_acc(last 250)="
                  f"{np.mean(accs[-250:]):.3f}  ({dt:.1f}s elapsed)")
    return model


if __name__ == "__main__":
    task = MQARTask(n_vocab=N_VOCAB, seed=SEED)
    gaps = [(20, 40), (60, 90), (150, 200), (300, 400)]

    runs = [
        ("oracle_decay", 0.95,  "oracle-timed, decay=0.95 (baseline)"),
        ("oracle_decay", 0.99,  "oracle-timed, decay=0.99"),
        ("oracle_decay", 0.999, "oracle-timed, decay=0.999"),
        ("learned",      0.95,  "learned gate, decay=0.95 (baseline)"),
        ("learned",      0.99,  "learned gate, decay=0.99"),
        ("learned",      0.999, "learned gate, decay=0.999"),
    ]

    models = {}
    for gate_mode, decay, label in runs:
        print(f"\n=== Training: {label} ===")
        m = train(task, gate_mode, decay, label=label)
        models[label] = m

    print("\n=== Held-out evaluation (gap 20-40, matches training) ===")
    for label, m in models.items():
        acc = evaluate(m)
        print(f"  {label:36s}  held-out = {acc:.3f}")

    print("\n=== Generalization vs gap length ===")
    print(f"  {'model':36s}  " + "  ".join(f"gap~{g[0]}-{g[1]:<4d}" for g in gaps))
    for label, m in models.items():
        row = [evaluate(m, n_episodes=100, gap_len=g) for g in gaps]
        print(f"  {label:36s}  " + "  ".join(f"{a:10.3f}" for a in row))

    print("\n=== Gate diagnostic (learned-gate models only) ===")
    for label, m in models.items():
        if "learned gate" in label:
            key_g, filler_g = gate_diagnostic(m)
            ratio = key_g / max(filler_g, 1e-6)
            print(f"  {label:36s}  key={key_g:.3f}  filler={filler_g:.3f}  ratio={ratio:.2f}x")
