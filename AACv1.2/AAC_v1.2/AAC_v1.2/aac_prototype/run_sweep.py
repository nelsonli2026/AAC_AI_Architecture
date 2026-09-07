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
SEED = 0  # fixed weight-init AND fixed fresh task per config -- every config in
# this grid sees the exact same episode stream, so differences reflect the
# hyperparameter, not sampling luck (within the limits of single-seed noise --
# see PHASE3_GATE_RESULTS.md's multi-seed finding; this grid is a screening
# tool, not a precision measurement).


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


def train_and_eval(aux_gate_weight, aux_pos_weight, n_episodes=N_EPISODES):
    task = MQARTask(n_vocab=N_VOCAB, seed=SEED)
    model = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=SEED,
                          gate_mode="learned", decay=0.999,
                          aux_gate_weight=aux_gate_weight, aux_pos_weight=aux_pos_weight)
    model.opt = AdamTensor(model.params, lr=LR)
    accs = []
    model.opt.zero_grad()
    for ep in range(1, n_episodes + 1):
        episode = task.sample_episode(n_pairs=6, n_queries=6)
        loss, acc = model.forward_episode(episode, train=True)
        accs.append(acc)
        if ep % BATCH == 0:
            model.opt.step()
            model.opt.zero_grad()
    held_out = evaluate(model, seed=999)
    key_g, filler_g = gate_diagnostic(model, seed=777)
    ratio = key_g / max(filler_g, 1e-6)
    return dict(aux_gate_weight=aux_gate_weight, aux_pos_weight=aux_pos_weight,
                held_out=held_out, ratio=ratio, key_gate=key_g, filler_gate=filler_g,
                final_train_acc=float(np.mean(accs[-250:])))


if __name__ == "__main__":
    pos_weights = [2, 5, 10, 20]
    gate_weights = [0.5, 1, 3]

    results = []
    t0 = time.perf_counter()
    for gw in gate_weights:
        for pw in pos_weights:
            r = train_and_eval(gw, pw)
            results.append(r)
            print(f"  gate_w={gw:4.1f}  pos_w={pw:4.1f}  held_out={r['held_out']:.3f}  "
                  f"ratio={r['ratio']:.2f}x  final_train_acc={r['final_train_acc']:.3f}  "
                  f"({time.perf_counter()-t0:.1f}s elapsed)")

    print("\n=== Full grid, sorted by held-out accuracy ===")
    for r in sorted(results, key=lambda r: -r["held_out"]):
        print(f"  gate_w={r['aux_gate_weight']:4.1f}  pos_w={r['aux_pos_weight']:4.1f}  "
              f"held_out={r['held_out']:.3f}  ratio={r['ratio']:.2f}x")

    headline = next(r for r in results if r["aux_gate_weight"] == 1 and r["aux_pos_weight"] == 5)
    best = max(results, key=lambda r: r["held_out"])
    print(f"\nHeadline config (gate_w=1, pos_w=5): held_out={headline['held_out']:.3f}")
    print(f"Best in grid (gate_w={best['aux_gate_weight']}, pos_w={best['aux_pos_weight']}): "
          f"held_out={best['held_out']:.3f}")
