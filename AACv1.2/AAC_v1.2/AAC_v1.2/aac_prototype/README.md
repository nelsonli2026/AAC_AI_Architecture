# AAC Prototype

A working, runnable implementation of the AAC (Adaptive Associative
Computing) architecture described in `AAC.txt` (see the top-level
`README.md` for the conceptual overview), incorporating the corrections
from `AAC-corrections.md` (non-circular update order, sign-corrected
credit assignment, de-collided notation).

**Pure NumPy, zero dependencies.** No PyTorch was available in this
environment (no network access to install it), so this is a from-scratch
implementation, including a hand-derived MLP + Adam optimizer for Layer 1,
and a hand-rolled reverse-mode autodiff engine for Layer 2. `pip install
numpy` is the only requirement.

## v1.2 status (read this first)

This project has gone through three phases of investigation, and the
headline finding changed direction more than once. **If you only read one
document, read `PHASE3_GATE_RESULTS.md` in full, including its later
sections** -- it supersedes claims made in `PHASE2_RESULTS.md` and
`DELTA_RULE_RESULTS.md`, and its own early sections are in turn superseded
by its later ones. Short version, as of the end of Phase 3:

- The differentiable relaxation (Layer 2, `aac/models_diff.py`) loses to
  plain softmax attention on the standard MQAR benchmark when both are
  trained under a constant learning rate (Phase 2's original setup).
- That gap was chased through: fixed decay bottleneck (solved), gate
  write-selectivity bottleneck (the model doesn't learn to write
  selectively) -- multiple unsupervised fixes failed, one supervised fix
  (an auxiliary BCE loss on the gate) worked, sort of, but with alarmingly
  high seed-to-seed variance.
- Chasing *that* variance down overturned the entire selectivity framing:
  the dominant cause was a constant, unscheduled Adam learning rate used
  throughout the whole investigation. Fixing the learning-rate schedule
  alone -- with zero gate supervision -- recovers most of the accuracy
  gain. Gate supervision still helps, but as a secondary +5-9 percentage
  point effect, not "the" fix.
- None of the LR-decay numbers are confirmed converged, and they turned
  out to be sensitive to the exact schedule duration used. **Nothing in
  this repository should currently be cited as a finished result.** The
  single most useful next thing to run is listed at the end of
  `PHASE3_GATE_RESULTS.md`.
- Along the way, three real bugs were found and fixed in the autodiff
  engine itself (a reference-cycle memory leak, a recursion-depth crash
  on long sequences, and a non-deterministic child-iteration order that
  made "same-seed reruns" not actually reproducible). All three are fixed
  in this version; see `PHASE3_GATE_RESULTS.md`'s Methods note for detail
  and `test_autodiff.py` for the regression checks.

If you're picking this project up fresh, the practical takeaway is: use
this version (v1.2) of `aac/autodiff.py` and `aac/models_diff.py`
regardless of which phase's script you're running, and treat every
accuracy number in `PHASE2_RESULTS.md` / `DELTA_RULE_RESULTS.md` /
`PHASE3_GATE_RESULTS.md`'s earlier sections as provisional history, not
current fact.

## Run it

```bash
# Layer 1 -- original, non-differentiable prototype
python3 train.py              # train + evaluate on synthetic associative recall
python3 mechanism_checks.py   # isolated checks: routing speed, merge, eviction
python3 scaling_sweep.py      # routing-scaling fit + noise-gap generalization

# Layer 2 -- differentiable relaxation (Phase 2/3, MQAR)
python3 test_autodiff.py      # gradient + determinism checks -- run this first
python3 train_compare.py      # RNN vs attention vs AAC-diff, matched budget
python3 run_oracle_vs_learned.py  # Phase 3 Finding 1
python3 run_gate_fixes.py         # Phase 3 Finding 3 (warm-start, curriculum)
python3 run_decay_sweep.py        # Phase 3 Findings 2+3 (decay x gate_mode)
python3 run_aux_gate.py           # Phase 3 Finding 4+5 (aux BCE supervision)
python3 run_multiseed.py          # Phase 3 robustness check (5-seed replication)
python3 run_sweep.py              # Phase 3 robustness check (hyperparameter grid)
```

## What Layer 1 demonstrates

`train.py` trains on a synthetic **associative-recall** task: the model is
shown 5 `(KEY_i, VAL_j)` pairs separated by noise tokens, then a long noise
gap (much longer than the fast state's effective memory), then queried on
each `KEY_i` and must output the correct `VAL_j`. This is the standard
stress test for "does this architecture actually use long-range memory, or
is it just pattern-matching locally."

Actual output from a full run:

```
episode   100  train_acc(last 100)=0.404  writes=5  promotions=5  evictions=0  final|P|=5
episode  1500  train_acc(last 100)=0.944  writes=5  promotions=5  evictions=0  final|P|=5

accuracy with persistent memory (routing on):   0.939
accuracy with persistent memory (routing off):  0.939
accuracy with memory READS DISABLED (ablation): 0.048
  -> memory contributes +0.891 accuracy (89.1 pts)
```

Chance accuracy is 1/16 ≈ 0.0625 (16 possible values). Three things are
worth noting:

1. **Learning happens** — accuracy climbs from ~40% to ~94% as the
   controller's utility estimator and the readout head train.
2. **Memory stays sparse** — exactly 5 items get promoted for 5 pairs, no
   more, no duplicates. `|P_t| ≪ N` (section 17) holds without being
   hard-coded — it falls out of the write-gating and merge logic.
3. **Memory is load-bearing** — forcibly blinding memory reads (monkey-
   patching `PersistentMemory.read` to return zeros) collapses accuracy to
   near-chance.

`mechanism_checks.py` isolates three mechanisms that don't show up
clearly at the small scale above:

- **Hierarchical routing** (section 11): at `|P_t| = 4000`, k-means-routed
  retrieval is ~2.7x faster than flat search over all items.
- **Merge vs. competing hypotheses** (section 16): a near-duplicate key
  with the *same* value merges into one higher-confidence item; a
  near-duplicate key with a *different* value is kept as a **separate**
  item rather than silently overwritten.
- **Eviction** (section 15): deliberately weak items (`conf=0.1`) decay
  below threshold and get removed.

`scaling_sweep.py` fits a log-log slope to routing latency vs. memory
size. The **fixed** cluster count used elsewhere (`n_clusters=8`) gives
slope ≈ 0.65 -- real speedup over flat search, but closer to linear
(`O(M)`) than the `O(log M)` the spec targets, since each cluster holds
`M/8` items regardless of `M`. Scaling the cluster count with `sqrt(M)`
brings the slope to ≈ 0.36, meaningfully sub-linear but still short of
true `O(log M)`.

## What Layer 2 demonstrates (and why it's more complicated than it looks)

See `PHASE2_RESULTS.md`, `DELTA_RULE_RESULTS.md`, and especially
`PHASE3_GATE_RESULTS.md`, read in that order, for the full investigation.
Layer 2 adds a hand-rolled, gradient-checked autodiff engine
(`aac/autodiff.py`), a harder shared-vocabulary recall task with no
pre-aligned embeddings (`task_mqar.py`), and a matched-budget comparison
against an RNN-only baseline and a softmax-attention baseline
(`aac/models_diff.py`, `train_compare.py`). The short version has changed
across phases -- see the "v1.2 status" section above rather than trusting
any single number quoted here.

## What's faithful to the spec vs. simplified (Layer 1)

**Faithful:**
- The `Z_t = (S_t, T_t, P_t)` three-timescale split, with the corrected
  (non-circular) per-step ordering.
- Structured fast-state transition `A_t = D_t + L_t Q_t^⊤` (renamed from
  `P_t` to `L_t` per the notation fix).
- Memory item schema `(k_i, v_i, c_i, t_i, E_i)`, sparse write gating,
  decaying temporary trace with evidence-based promotion, associative
  softmax read, hierarchical (k-means) routing, confidence reinforcement
  and eviction, compatible-merge vs. competing-hypothesis preservation.
- Ablation-based causal credit assignment with the **corrected sign**:
  `C_i = L_ablated(i) − L_base` (positive when the memory genuinely helps).
- The value-of-computation gate (learned utility vs. fixed cost threshold)
  and adaptive multi-step reasoning driven by output-entropy.

**Simplified, and why:**
- **No end-to-end backprop through time in Layer 1.** `S_t`'s weights are
  a fixed random ("echo state network") reservoir rather than learned via
  BPTT. Only the controller's utility MLP and the output readout MLP are
  trained. Layer 2 is the from-scratch attempt at doing this properly.
- **Token identity embeddings are fixed, not learned** in Layer 1's task.
  Layer 2's MQAR task removes this simplification entirely.
- **Utility estimator is trained with a proxy label** (`1` for KEY/VAL
  events, `0` for filler/query), not the true counterfactual `U(e_t)`
  from section 7. The ablation-based estimator in `memory.py` *does*
  compute the real counterfactual, but only as a periodic pass.
- **One memory per episode**, reset at the start of each training episode.

## File map

```
task.py                 Layer 1: synthetic associative-recall episode generator
aac/mlp.py               Layer 1: hand-rolled MLP + Adam (only gradient-trained part)
aac/fast_state.py        Layer 1: S_t, diagonal + low-rank structured transition
aac/trace.py             Layer 1: T_t, decaying evidence buffer, promotion
aac/memory.py            Layer 1: P_t, write/read/routing/reinforce/evict/merge
aac/controller.py        Layer 1: learned utility estimator + value-of-computation gating
aac/model.py             Layer 1: wires it all together, non-circular update order
train.py                Layer 1: training loop, held-out eval, ablation demo
mechanism_checks.py     Layer 1: isolated routing/merge/eviction checks
scaling_sweep.py        Layer 1: log-log routing-scaling fit + noise-gap sweep

aac/autodiff.py          Layer 2: hand-rolled reverse-mode autodiff (Tensor + ops)
test_autodiff.py        Layer 2: numerical gradient checks + determinism check
task_mqar.py             Layer 2: harder MQAR/induction-head task, shared vocab
aac/models_diff.py       Layer 2: RNNBaseline, AttentionBaseline, AACDiffModel
train_compare.py        Layer 2: Phase 2 comparison harness, gate diagnostics
run_oracle_vs_learned.py    Phase 3 Finding 1 (read mechanism isolation)
run_gate_fixes.py           Phase 3 Finding 3 (warm-start, curriculum)
run_decay_sweep.py          Phase 3 Findings 2+3 (decay x gate_mode grid)
run_aux_gate.py              Phase 3 Findings 4+5 (aux BCE supervision)
run_multiseed.py             Phase 3 robustness check (5-seed replication)
run_sweep.py                 Phase 3 robustness check (hyperparameter grid)

PHASE2_RESULTS.md        baseline comparison, matched-budget results (superseded in part)
DELTA_RULE_RESULTS.md    delta-rule variants, off-by-one gate-diagnostic fix
PHASE3_GATE_RESULTS.md   the full gate-selectivity / LR-schedule investigation (read this)
```
