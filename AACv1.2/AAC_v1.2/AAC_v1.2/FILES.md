# FILES.md — Project Map (v1.2)

This is the consolidated, deduplicated successor to the three overlapping
`AAC v1.1` folder trees (`aac_prototype (2)/aac_prototype/`,
`aac_prototype (2)/New folder/aac_prototype/`, and `AAC-PROT/...`). Those
are no longer needed once this folder is in place — everything in them is
either superseded here or was an exact duplicate.

## ⚠️ Read `aac_prototype/PHASE3_GATE_RESULTS.md` before trusting any number in this repo

The project's headline finding changed direction twice during Phase 3.
`PHASE3_GATE_RESULTS.md`'s own early sections are superseded by its later
ones ("Isolating the LR-decay effect from gate supervision" is the
current state of understanding). `PHASE2_RESULTS.md` and
`DELTA_RULE_RESULTS.md` are both partially superseded by it too. Every
one of those three files now carries a short note at the top pointing
this out — but the short version is: nothing in this repository should
currently be cited as a finished, converged result.

---

## Layer 1 — the original, non-differentiable prototype

Trained via hand-rolled MLP + Adam only; the memory system itself
(write/promote/evict/merge/reinforce) uses explicit algorithmic rules,
not backprop. Entirely independent of Layer 2 below — no shared code.

| File | Role |
|---|---|
| `aac_prototype/task.py` | `AssociativeRecallTask` — the original synthetic task, disjoint KEY/VAL/QUERY/FILLER token ranges, privileged identity embeddings. |
| `aac_prototype/aac/mlp.py` | Hand-derived 2-layer MLP + Adam. The only gradient-trained component in this layer. |
| `aac_prototype/aac/fast_state.py` | `FastState` (`S_t`) — fixed random reservoir (echo-state style), not trained. |
| `aac_prototype/aac/trace.py` | `TemporaryTrace` (`T_t`) — decaying evidence buffer, promotion logic. |
| `aac_prototype/aac/memory.py` | `PersistentMemory` (`P_t`) — write/read/routing/reinforce/evict/merge/ablation credit-assignment. |
| `aac_prototype/aac/controller.py` | Learned utility estimator (small MLP) + value-of-computation gating. |
| `aac_prototype/aac/model.py` | `AACModel` — wires all of the above together, non-circular update order per `AAC-CORRECTIONS.md`. |
| `aac_prototype/train.py` | Trains `AACModel` on `task.py`, reports accuracy + memory-ablation demo. |
| `aac_prototype/mechanism_checks.py` | Isolated checks: routing speed at scale, merge vs. competing-hypothesis, eviction. |
| `aac_prototype/scaling_sweep.py` | Routing-latency log-log scaling fit + noise-gap generalization sweep. |

---

## Layer 2 — the differentiable relaxation (Phase 2 / Phase 3, MQAR)

Full end-to-end BPTT via a hand-rolled autodiff engine.

| File | Role |
|---|---|
| `aac_prototype/aac/autodiff.py` | The autodiff engine: `Tensor`, and every op. **v1.2 state**: includes `bce_loss`; the reference-cycle memory leak in `backward()` is fixed; `free_graph()` handles eval-mode cleanup; the topological sort is iterative (`_topo_sort`, no recursion-depth limit); `_prev` is a deterministically-ordered list (creation-order id), not an identity-hashed `set` — same-seed reruns are now bit-identical across process invocations. |
| `aac_prototype/test_autodiff.py` | Numerical gradient checks for every op, **including `bce_loss`** and a same-process determinism check (new in v1.2 — both were previously validated ad hoc but not committed as regression tests). |
| `aac_prototype/task_mqar.py` | `MQARTask` — the harder, shared-vocabulary induction-head task used for all Phase 2/3 results. |
| `aac_prototype/aac/models_diff.py` | `RNNBaseline`, `AttentionBaseline`, `AACDiffModel`. Supports `gate_mode` in `{"learned", "oracle", "oracle_decay", "curriculum"}`, plus `gate_bias_init`, `aux_gate_weight`, `aux_pos_weight` for the Phase 3 auxiliary-supervision experiments. |
| `aac_prototype/train_compare.py` | The Phase 2 / delta-rule-phase comparison harness: trains all three models, held-out eval, gap-length generalization sweep, gate diagnostic, perfect-write test. Carries a docstring note on which of its own conclusions Phase 3 revises. |
| `aac_prototype/PHASE2_RESULTS.md` | Phase 2 write-up. Partially superseded — see the note at its top. |
| `aac_prototype/DELTA_RULE_RESULTS.md` | Delta-rule variants, the off-by-one gate-diagnostic bug fix. Partially superseded — see the note at its top. |
| `aac_prototype/PHASE3_GATE_RESULTS.md` | **The current state of the investigation.** Gate-selectivity findings, the robustness check that found the results didn't replicate, and the LR-schedule discovery that overturned the phase's own central claim. |

**Phase 3 experiment scripts** (each corresponds to a section of `PHASE3_GATE_RESULTS.md`):

| File | Corresponds to |
|---|---|
| `aac_prototype/run_oracle_vs_learned.py` | Finding 1 (read mechanism isolation) |
| `aac_prototype/run_gate_fixes.py` | Finding 3 (warm-start, curriculum) |
| `aac_prototype/run_decay_sweep.py` | Findings 2+3 (decay × gate_mode grid) |
| `aac_prototype/run_aux_gate.py` | Findings 4+5 (auxiliary BCE gate supervision) |
| `aac_prototype/run_multiseed.py` | Robustness check (5-seed replication) |
| `aac_prototype/run_sweep.py` | Robustness check (hyperparameter grid) |

The LR-decay isolation experiments (the no-aux control, the 9000-episode
convergence checks) were run as inline scratch scripts during the
investigation rather than saved as standalone files — the exact
methodology is fully described in `PHASE3_GATE_RESULTS.md`'s "Isolating
the LR-decay effect from gate supervision" section and is a
straightforward modification of `run_aux_gate.py`'s training loop (add
`aux_gate_weight=0` for the no-aux control; add an `lr_schedule(ep,
n_total)` function and set `model.opt.lr` each episode for the decay).
Reconstructing them as standalone scripts is reasonable follow-up work if
this investigation continues.

---

## Quick nav — "I want to..."

| Task | Files to touch |
|---|---|
| Check the engine is trustworthy before changing anything | `python3 aac_prototype/test_autodiff.py` — should print `ALL PASS` |
| Run the current gate-selectivity / LR-schedule experiments | `run_aux_gate.py`, `run_multiseed.py`, `run_sweep.py` (needs `aac/autodiff.py` + `aac/models_diff.py`) |
| Add a new autodiff op | `aac/autodiff.py`, then add a gradient check for it in `test_autodiff.py` before using it anywhere — this is the project's own stated norm, and it was skipped once for `bce_loss` before being fixed in v1.2 |
| Change the MQAR task (episode structure, vocab size, gap distribution) | `task_mqar.py` |
| Reproduce or extend the Phase 2 baseline comparison | `train_compare.py` + `aac/models_diff.py` + `task_mqar.py` |
| Run the single most important next experiment in this project | See the last section of `PHASE3_GATE_RESULTS.md` — as of v1.2, that's confirming convergence and testing the LR-decay effect on the Phase 2 attention baseline, which has never been given the same optimizer treatment |
| A quick one-off experiment/ablation | Copy `run_aux_gate.py`'s pattern — minimal deps are `task_mqar.py` + `aac/models_diff.py` + `aac/autodiff.py` |
| Work on the original (non-diff) architecture instead | `train.py` + `aac/model.py` + `aac/memory.py` + `aac/controller.py` + `aac/trace.py` + `aac/fast_state.py` + `task.py` — entirely separate from everything in Layer 2 |
| API | **There isn't one.** This is a local research prototype (`uv run script.py`), not a served API. |

---

## Cross-cutting docs (apply to the whole project)

| File | Role |
|---|---|
| `README.md` | Top-level conceptual overview of the AAC architecture. |
| `AAC-CORRECTIONS.md` | The three math corrections to the original spec (circular dependency, credit-assignment sign error, notation collisions) — referenced throughout Layer 1's code comments. |
| `LICENSE` | Apache 2.0. |

## What changed from v1.1 to v1.2

- Three duplicate folder trees consolidated into one.
- Three autodiff engine bugs fixed: reference-cycle memory leak, recursion-depth crash on long sequences, non-deterministic child-iteration order (see `PHASE3_GATE_RESULTS.md` Methods note for full detail on each).
- `test_autodiff.py` extended with a `bce_loss` gradient check and a determinism check — both existed only as ad hoc scratch verification in v1.1.
- `PHASE3_GATE_RESULTS.md` added, documenting the full gate-selectivity investigation, the robustness check that overturned its own headline number, and the subsequent discovery that an unscheduled learning rate — not gate selectivity — was the dominant effect in the entire phase.
- Provenance notes added to `PHASE2_RESULTS.md`, `DELTA_RULE_RESULTS.md`, and `train_compare.py` pointing to where Phase 3 revises their conclusions.
