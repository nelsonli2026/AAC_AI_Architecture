"""
Train the AAC prototype on the synthetic associative-recall task and
report the metrics that actually demonstrate the architecture's claims:

  - task accuracy over training (does the controller learn to gate writes
    usefully, and does the readout learn to use retrieved memory?)
  - memory traffic stats (writes / promotions / evictions / merges / final
    |P_t|) -- should show |P_t| << sequence length (section 17)
  - an ablation with memory reads disabled entirely, to show the
    persistent memory is *load-bearing* and not decorative
  - a live demonstration of the ablation-based credit-assignment update
    (section 21, sign corrected) on a held-out episode
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from task import AssociativeRecallTask
from aac.model import AACModel


def run(n_episodes=1500, eval_every=100, seed=0):
    task = AssociativeRecallTask(n_ids=16, n_filler_vocab=40, seed=seed)
    model = AACModel(n_ids=16, n_filler_vocab=40, seed=seed)

    accs, mem_sizes = [], []
    for ep in range(1, n_episodes + 1):
        episode = task.sample_episode(n_pairs=5, n_queries=5)
        stats, memory, qlog, S_prev = model.run_episode(episode, train=True)
        accs.append(stats["accuracy"])
        mem_sizes.append(stats["final_memory_size"])

        if ep % eval_every == 0:
            recent_acc = np.mean(accs[-eval_every:])
            recent_mem = np.mean(mem_sizes[-eval_every:])
            print(f"episode {ep:5d}  train_acc(last {eval_every})="
                  f"{recent_acc:.3f}  writes={stats['n_writes']:3d} "
                  f"promotions={stats['n_promotions']:3d} "
                  f"evictions={stats['n_evictions']:3d} "
                  f"merges={stats['n_merges']:3d} "
                  f"final|P|={stats['final_memory_size']:3d} "
                  f"(seq_len={stats['seq_len']})")

    return task, model, accs, mem_sizes


def evaluate(task, model, n_episodes=200, use_routing=True, seed=999):
    rng_task = AssociativeRecallTask(n_ids=16, n_filler_vocab=40, seed=seed)
    accs = []
    for _ in range(n_episodes):
        episode = rng_task.sample_episode(n_pairs=5, n_queries=5)
        stats, _, _, _ = model.run_episode(episode, train=False,
                                            use_routing=use_routing)
        accs.append(stats["accuracy"])
    return float(np.mean(accs))


class _NoMemory:
    """Wraps model.controller / memory read to simulate "memory disabled":
    monkeypatch PersistentMemory.read to always return zeros, isolating
    how much of the model's accuracy actually comes from persistent
    memory vs. the fast state + readout alone."""


def evaluate_without_memory(task, model, n_episodes=200, seed=999):
    from aac import memory as memory_mod
    original_read = memory_mod.PersistentMemory.read

    def blind_read(self, queries, top_k=4, use_routing=True):
        return np.zeros(self.d_val), []

    memory_mod.PersistentMemory.read = blind_read
    try:
        acc = evaluate(task, model, n_episodes=n_episodes, seed=seed)
    finally:
        memory_mod.PersistentMemory.read = original_read
    return acc


def ablation_demo(task, model, seed=12345):
    ep = AssociativeRecallTask(n_ids=16, n_filler_vocab=40, seed=seed) \
        .sample_episode(n_pairs=5, n_queries=5)
    stats, memory, qlog, S_prev = model.run_episode(
        ep, train=False, collect_ablation=True)
    print(f"\n[ablation demo] episode accuracy={stats['accuracy']:.2f}, "
          f"final |P_t|={len(memory)}")
    result = model.ablation_study(memory, qlog, S_prev, n_items_to_test=5)
    if result is None:
        print("  (not enough memory items / queries to run ablation demo)")
        return
    print("  idx | confidence before -> after  (C_i = L_ablated - L_base, "
          "sign-corrected section 21)")
    for idx, before, after in result:
        direction = "reinforced" if after > before else "weakened"
        print(f"   {idx:3d} | {before:6.3f} -> {after:6.3f}   ({direction})")


if __name__ == "__main__":
    print("=== Training AAC prototype on synthetic associative recall ===\n")
    task, model, accs, mem_sizes = run(n_episodes=1500, eval_every=100)

    print("\n=== Held-out evaluation ===")
    acc_full = evaluate(task, model, n_episodes=200)
    print(f"accuracy with persistent memory (routing on):  {acc_full:.3f}")

    acc_no_routing = evaluate(task, model, n_episodes=200, use_routing=False)
    print(f"accuracy with persistent memory (routing off): {acc_no_routing:.3f}")

    acc_blind = evaluate_without_memory(task, model, n_episodes=200)
    print(f"accuracy with memory READS DISABLED (ablation): {acc_blind:.3f}")
    print(f"  -> memory contributes {acc_full - acc_blind:+.3f} accuracy "
          f"({(acc_full - acc_blind) * 100:.1f} pts)")

    ablation_demo(task, model)
