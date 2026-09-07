"""
Two sweeps that push past the single-point checks in mechanism_checks.py:

1. ROUTING SCALING: does retrieval latency actually grow like O(log M), as
   the spec claims, or closer to O(M)? mechanism_checks.py uses a FIXED
   cluster count (n_clusters=8), which means each cluster holds ~M/8
   items -- local search inside a cluster is then O(M/8), i.e. still
   *linear* in M, just with a smaller constant. This sweep fits a log-log
   slope to check that directly, and compares against a variant where the
   cluster count scales with sqrt(M) (a cheap approximation of true
   hierarchical routing), which should flatten the curve further.

   Interpreting the fitted slope:
     slope ~ 1.0   -> effectively O(M)   (linear)
     slope ~ 0.5   -> close to O(sqrt(M))
     slope ~ 0     -> close to O(log M) or O(1) (flat)

2. GAP-LENGTH SWEEP: train.py always uses a moderate noise gap
   (gap_len=(40,60)). This sweep evaluates the ALREADY-TRAINED model at
   much longer gaps to find where the fast-state-only (memory-blind)
   baseline actually collapses, versus how well the full model holds up.
"""
import sys, time
import numpy as np

sys.path.insert(0, ".")
from aac.memory import PersistentMemory
from task import AssociativeRecallTask
from aac.model import AACModel


# ----------------------------------------------------------------------
# 1. Routing scaling sweep
# ----------------------------------------------------------------------
def _time_reads(mem, queries, use_routing):
    t0 = time.perf_counter()
    for q in queries:
        mem.read([q], top_k=4, use_routing=use_routing)
    return time.perf_counter() - t0


def routing_scaling_sweep(sizes=(500, 1000, 2000, 4000, 8000, 16000, 32000),
                           d=16, n_queries=200, seed=0):
    rows = []
    for M in sizes:
        rng = np.random.default_rng(seed)
        mem = PersistentMemory(d, d, max_items=M + 10, seed=seed)
        for i in range(M):
            mem.write(rng.normal(size=d), rng.normal(size=d), t_now=i,
                       initial_conf=1.0)
        queries = rng.normal(size=(n_queries, d))

        t_flat = _time_reads(mem, queries, use_routing=False)

        mem.rebuild_routing()  # fixed n_clusters (default 8)
        t_fixed = _time_reads(mem, queries, use_routing=True)

        k_sqrt = max(2, int(np.sqrt(M)))
        mem.rebuild_routing(n_clusters=k_sqrt)
        t_sqrt = _time_reads(mem, queries, use_routing=True)

        rows.append((M, t_flat, t_fixed, t_sqrt, k_sqrt))
        print(f"  M={M:6d}  flat={t_flat/n_queries*1e6:8.1f} us/q  "
              f"fixed-k8={t_fixed/n_queries*1e6:8.1f} us/q  "
              f"sqrt-k{k_sqrt:3d}={t_sqrt/n_queries*1e6:8.1f} us/q")

    Ms = np.array([r[0] for r in rows], dtype=float)
    logM = np.log(Ms)

    def slope(col_idx):
        y = np.log(np.array([r[col_idx] for r in rows]))
        s, _ = np.polyfit(logM, y, 1)
        return s

    print("\n  log-log fit (latency ~ M^slope):")
    print(f"    flat search:         slope = {slope(1):.2f}  "
          f"(expect ~1.0, i.e. linear)")
    print(f"    routed, fixed k=8:   slope = {slope(2):.2f}  "
          f"(fixed cluster count -> still ~linear, smaller constant)")
    print(f"    routed, k=sqrt(M):   slope = {slope(3):.2f}  "
          f"(cluster count grows with M -> should be flatter, closer to "
          f"sqrt(M) scaling)")
    return rows


# ----------------------------------------------------------------------
# 2. Gap-length sweep (uses a freshly trained model)
# ----------------------------------------------------------------------
def train_small_model(n_episodes=1500, seed=0):
    task = AssociativeRecallTask(n_ids=16, n_filler_vocab=40, seed=seed)
    model = AACModel(n_ids=16, n_filler_vocab=40, seed=seed)
    for _ in range(n_episodes):
        episode = task.sample_episode(n_pairs=5, n_queries=5)
        model.run_episode(episode, train=True)
    return task, model


def eval_at_gap(model, gap, n_episodes=100, seed=999):
    from aac import memory as memory_mod
    eval_task = AssociativeRecallTask(n_ids=16, n_filler_vocab=40, seed=seed)

    accs_full, accs_blind = [], []
    original_read = memory_mod.PersistentMemory.read

    def blind_read(self, queries, top_k=4, use_routing=True):
        return np.zeros(self.d_val), []

    for _ in range(n_episodes):
        episode = eval_task.sample_episode(n_pairs=5, n_queries=5,
                                            gap_len=(gap, gap + 10))
        stats, *_ = model.run_episode(episode, train=False)
        accs_full.append(stats["accuracy"])

        memory_mod.PersistentMemory.read = blind_read
        try:
            stats_b, *_ = model.run_episode(episode, train=False)
        finally:
            memory_mod.PersistentMemory.read = original_read
        accs_blind.append(stats_b["accuracy"])

    return float(np.mean(accs_full)), float(np.mean(accs_blind))


def gap_length_sweep(gaps=(10, 40, 100, 250, 500, 1000)):
    print("\n(training a model once at the default gap length, then "
          "evaluating it at longer and longer gaps it never saw in "
          "training)")
    task, model = train_small_model(n_episodes=1500)
    print(f"\n  {'gap':>6} | {'full model':>10} | {'memory-blind':>12} | gap")
    for g in gaps:
        acc_full, acc_blind = eval_at_gap(model, g)
        print(f"  {g:6d} | {acc_full:10.3f} | {acc_blind:12.3f} | "
              f"{'<-- fails' if acc_blind < 0.2 and acc_full > 0.5 else ''}")
    return


if __name__ == "__main__":
    print("=== 1. Routing scaling sweep ===")
    routing_scaling_sweep()

    print("\n=== 2. Gap-length sweep (generalization beyond training gap) ===")
    gap_length_sweep()
