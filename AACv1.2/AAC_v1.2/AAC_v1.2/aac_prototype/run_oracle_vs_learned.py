import sys, time
import numpy as np

sys.path.insert(0, ".")
from task_mqar import MQARTask
from aac.models_diff import RNNBaseline, AACDiffModel, AdamTensor

D_EMB, D_H = 32, 32
N_VOCAB = 64
SEED = 0
LR = 1e-2
BATCH = 8
N_EPISODES = 2000


def train_model(model, task, n_episodes, eval_every=250, lr=LR, batch=BATCH, label=""):
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
            print(f"  [{label:30s}] ep {ep:5d}  train_acc(last {eval_every})="
                  f"{np.mean(accs[-eval_every:]):.3f}  ({dt:.1f}s elapsed)")
    return accs


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


if __name__ == "__main__":
    task = MQARTask(n_vocab=N_VOCAB, seed=SEED)

    print("=== Training RNN-only baseline (floor) ===")
    rnn = RNNBaseline(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=SEED)
    train_model(rnn, task, N_EPISODES, label="RNN-only")

    print("\n=== Training AAC-diff, LEARNED gate (normal) ===")
    learned = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=SEED,
                            update_rule="hebbian", gate_mode="learned")
    train_model(learned, task, N_EPISODES, label="AAC-diff (learned gate)")

    print("\n=== Training AAC-diff, ORACLE gate (perfect write timing given for free) ===")
    oracle = AACDiffModel(vocab_size=N_VOCAB, d_emb=D_EMB, d_h=D_H, seed=SEED,
                           update_rule="hebbian", gate_mode="oracle")
    train_model(oracle, task, N_EPISODES, label="AAC-diff (oracle gate)")

    print("\n=== Held-out evaluation ===")
    acc_rnn = evaluate(rnn)
    acc_learned = evaluate(learned)
    acc_oracle = evaluate(oracle)
    print(f"  RNN-only:            {acc_rnn:.3f}")
    print(f"  AAC-diff (learned):  {acc_learned:.3f}")
    print(f"  AAC-diff (oracle):   {acc_oracle:.3f}")
    print(f"  chance = {1/N_VOCAB:.3f}")

    print("\n=== Generalization vs gap length ===")
    gaps = [(20, 40), (60, 90), (150, 200), (300, 400)]
    print(f"  {'model':22s}  " + "  ".join(f"gap~{g[0]}-{g[1]:<4d}" for g in gaps))
    for name, m in [("learned", learned), ("oracle", oracle)]:
        row = [evaluate(m, n_episodes=100, gap_len=g) for g in gaps]
        print(f"  {name:22s}  " + "  ".join(f"{a:10.3f}" for a in row))

    print("\n=== Gate diagnostic (learned model only -- oracle gate is hard-coded) ===")
    key_g, filler_g = gate_diagnostic(learned)
    print(f"  mean gate at KEY->VALUE write positions: {key_g:.3f}")
    print(f"  mean gate at FILLER positions:            {filler_g:.3f}")
    print(f"  ratio: {key_g / max(filler_g, 1e-6):.2f}x")

    print("\n=== INTERPRETATION ===")
    print(f"  learned - RNN floor         : {acc_learned - acc_rnn:+.3f}")
    print(f"  oracle  - learned           : {acc_oracle - acc_learned:+.3f}")
    print("  If oracle >> learned  -> the GATE is the bottleneck (write timing).")
    print("  If oracle ~= learned (both low) -> the READ mechanism is (also) broken,")
    print("  since oracle removes the gate-learning problem entirely and it still fails.")
