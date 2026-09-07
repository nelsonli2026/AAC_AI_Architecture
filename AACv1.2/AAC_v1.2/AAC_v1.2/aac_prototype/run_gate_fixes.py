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


def train_warmstart(task, bias, n_episodes=N_EPISODES, label=""):
    model = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=SEED,
                          gate_mode="learned", gate_bias_init=bias)
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
            print(f"  [{label:26s}] ep {ep:5d}  train_acc(last 250)="
                  f"{np.mean(accs[-250:]):.3f}  ({dt:.1f}s elapsed)")
    return model, accs


def train_curriculum(task, n_episodes=N_EPISODES, ramp_frac=0.6, label="curriculum"):
    """alpha ramps 0 -> 1 linearly over the first ramp_frac of training, then
    stays at 1 (pure learned gate) for the remainder, so we can see whether
    it holds onto the selectivity once the oracle scaffolding is removed."""
    model = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=SEED,
                          gate_mode="curriculum", gate_bias_init=0.0)
    model.opt = AdamTensor(model.params, lr=LR)
    accs, alphas = [], []
    t0 = time.perf_counter()
    model.opt.zero_grad()
    ramp_episodes = int(n_episodes * ramp_frac)
    for ep in range(1, n_episodes + 1):
        model.alpha = min(1.0, ep / max(1, ramp_episodes))
        alphas.append(model.alpha)
        episode = task.sample_episode(n_pairs=6, n_queries=6)
        loss, acc = model.forward_episode(episode, train=True)
        accs.append(acc)
        if ep % BATCH == 0:
            model.opt.step()
            model.opt.zero_grad()
        if ep % 250 == 0:
            dt = time.perf_counter() - t0
            print(f"  [{label:26s}] ep {ep:5d}  alpha={model.alpha:.2f}  "
                  f"train_acc(last 250)={np.mean(accs[-250:]):.3f}  ({dt:.1f}s elapsed)")
    model.alpha = 1.0  # ensure eval uses pure learned gate
    return model, accs


if __name__ == "__main__":
    task = MQARTask(n_vocab=N_VOCAB, seed=SEED)

    print("=== Baseline recap: learned gate, bias=0 (from previous run) ===")
    print("  held-out ~ 1.1%  (chance = 1.6%)")

    print("\n=== A. Warm-start: gate bias initialized OPEN (bg=+2.0, sigmoid~0.88) ===")
    model_warm, _ = train_warmstart(task, bias=2.0, label="warmstart bg=+2.0")

    print("\n=== B. Warm-start: gate bias initialized VERY OPEN (bg=+4.0, sigmoid~0.98) ===")
    model_warm2, _ = train_warmstart(task, bias=4.0, label="warmstart bg=+4.0")

    print("\n=== C. Curriculum: anneal oracle-write -> learned-write over 60% of training ===")
    model_curric, _ = train_curriculum(task, ramp_frac=0.6, label="curriculum")

    print("\n=== Held-out evaluation ===")
    results = {
        "warmstart bg=+2.0": model_warm,
        "warmstart bg=+4.0": model_warm2,
        "curriculum (alpha->1)": model_curric,
    }
    for name, m in results.items():
        acc = evaluate(m)
        print(f"  {name:26s}  held-out acc = {acc:.3f}   (chance=0.016, prev learned=0.011, prev oracle=0.988)")

    print("\n=== Gate diagnostic ===")
    for name, m in results.items():
        key_g, filler_g = gate_diagnostic(m)
        ratio = key_g / max(filler_g, 1e-6)
        print(f"  {name:26s}  key={key_g:.3f}  filler={filler_g:.3f}  ratio={ratio:.2f}x")

    print("\n=== Generalization vs gap length (best of the three) ===")
    best_name = max(results, key=lambda n: evaluate(results[n], n_episodes=100))
    best_model = results[best_name]
    print(f"  (using: {best_name})")
    gaps = [(20, 40), (60, 90), (150, 200), (300, 400)]
    for g in gaps:
        acc = evaluate(best_model, n_episodes=100, gap_len=g)
        print(f"  gap {g}: {acc:.3f}")
