"""
Focused checks for three mechanisms that don't show up clearly in the
5-item episodic memory used by train.py: hierarchical routing's speed
benefit at scale (section 11), compatible-memory merging (section 16),
and confidence-based eviction (section 15).
"""
import sys, time
import numpy as np

sys.path.insert(0, ".")
from aac.memory import PersistentMemory


def routing_speed_check(M=4000, d=16, n_queries=300):
    rng = np.random.default_rng(0)
    mem = PersistentMemory(d, d, max_items=M + 10, seed=0)
    for i in range(M):
        k = rng.normal(size=d)
        v = rng.normal(size=d)
        mem.write(k, v, t_now=i, initial_conf=1.0)
    mem.rebuild_routing()
    queries = rng.normal(size=(n_queries, d))

    t0 = time.perf_counter()
    for q in queries:
        mem.read([q], top_k=4, use_routing=True)
    t_routed = time.perf_counter() - t0

    t0 = time.perf_counter()
    for q in queries:
        mem.read([q], top_k=4, use_routing=False)
    t_flat = time.perf_counter() - t0

    print(f"[routing check] |P_t|={M}, {n_queries} queries")
    print(f"  flat search:    {t_flat*1000:7.1f} ms total "
          f"({t_flat/n_queries*1e6:6.1f} us/query)")
    print(f"  routed search:  {t_routed*1000:7.1f} ms total "
          f"({t_routed/n_queries*1e6:6.1f} us/query)")
    print(f"  speedup: {t_flat / max(t_routed, 1e-9):.2f}x")


def merge_check(d=16):
    rng = np.random.default_rng(1)
    mem = PersistentMemory(d, d, seed=1)
    k = rng.normal(size=d)
    v = rng.normal(size=d)
    mem.write(k, v, t_now=0, initial_conf=1.0)
    # near-duplicate key, same value -> should MERGE, not add a new item
    mem.write(k + rng.normal(scale=0.01, size=d), v, t_now=1, initial_conf=1.0)
    print(f"\n[merge check] compatible duplicate write")
    print(f"  items after 2 compatible writes: {len(mem)} (expect 1)")
    print(f"  merges recorded: {mem.n_merges} (expect 1)")
    print(f"  confidence of surviving item: {mem.c[0]:.2f} (expect ~2.0)")

    # same key region, DIFFERENT value -> competing hypothesis, should NOT merge
    v2 = -v  # deliberately different
    mem.write(k + rng.normal(scale=0.01, size=d), v2, t_now=2, initial_conf=1.0)
    print(f"  items after a contradictory write: {len(mem)} (expect 2 -- "
          f"kept as competing hypotheses, not overwritten)")


def eviction_check(d=16):
    rng = np.random.default_rng(2)
    mem = PersistentMemory(d, d, evict_threshold=0.05, seed=2)
    for i in range(10):
        mem.write(rng.normal(size=d), rng.normal(size=d), t_now=i,
                   initial_conf=0.1)  # deliberately weak
    print(f"\n[eviction check] wrote 10 weak items (conf=0.10 each)")
    print(f"  items before decay: {len(mem)}")
    for _ in range(20):
        mem.decay_all(eta_d=0.01)
    mem.evict()
    print(f"  items after repeated decay + evict(): {len(mem)} "
          f"(low-confidence items removed)")
    print(f"  evictions recorded: {mem.n_evictions}")


if __name__ == "__main__":
    routing_speed_check()
    merge_check()
    eviction_check()
