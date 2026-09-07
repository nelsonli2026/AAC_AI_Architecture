# AAC — Adaptive Associative Computing (research prototype)

> **Status: early-stage research investigation, not a validated architecture.**
> This repo documents an ongoing attempt to test whether AAC's core idea
> holds up, using progressively more rigorous experiments. The honest
> summary as of the most recent phase: **the central comparative claim
> this project has been built around (AAC-diff vs. attention) is
> currently unresolved**, because a methodology bug was found that
> invalidates the fair-comparison basis of every number produced so far.
> See "Current status" below before citing any number in this repo.

## What AAC is

AAC is a proposed neural sequence architecture organized around one
idea: not all information deserves the same computational treatment.
Instead of a single hidden state or full attention over the whole
history, AAC splits memory into three timescales —

```
Z_t = (S_t, T_t, P_t)
```

— a fast recurrent state (`S_t`), a temporary trace where uncertain
information accumulates evidence (`T_t`), and a sparse, persistent
associative memory for information judged durably valuable (`P_t`) —
governed by a learned controller that decides, at every step, whether
the expected value of remembering something exceeds the cost of storing
or retrieving it. The full specification (with derivations) is in
[`AAC.txt`](AAC.txt); [`AAC-corrections.md`](AAC-corrections.md)
documents math errors found and fixed in the original spec (a circular
dependency in the update order, a sign error in credit assignment,
symbol collisions) before anything below was built.

**This repo is the process of actually testing that idea, not a
finished implementation of it.** Read the phase documents in order —
each one either confirms, narrows, or overturns something from the one
before it, and the overturns are left visible rather than edited away.

## Current status (read this before anything else)

The investigation has three phases, run against progressively harder
tests:

| Phase | What it tested | Verdict |
|---|---|---|
| [`aac_prototype/`](aac_prototype/) (non-differentiable) | Do the discrete mechanisms (write/promote/evict/merge/hierarchical routing) behave as specified? | **Yes.** All mechanisms work and are stress-tested (`mechanism_checks.py`, `scaling_sweep.py`) — but writes/reads were hand-fed via pre-aligned identity embeddings and trained with proxy supervision, not learned end-to-end. |
| [`PHASE2_RESULTS.md`](PHASE2_RESULTS.md) | A *differentiable* relaxation of the memory (continuous gated fast-weight matrix, real backprop, no privileged embeddings) vs. an RNN baseline and a softmax-attention baseline, on a standard induction-head recall task | AAC-diff appeared to lose badly to attention (1.4% vs. 57.2%). **This comparison is now known to be confounded — see Phase 3.** |
| [`DELTA_RULE_RESULTS.md`](DELTA_RULE_RESULTS.md) | Does a DeltaNet-style write rule fix it? | No measurable effect at the time — but this was also run under the same confound found in Phase 3. |
| [`PHASE3_GATE_RESULTS.md`](PHASE3_GATE_RESULTS.md) | Systematically isolating *why* AAC-diff was failing | Found and fixed two real bugs (a testing artifact that manufactured a false "dual bottleneck," and a non-deterministic autodiff tie-break) — then found that **every prior experiment, including Phase 2's headline comparison, was run under a constant unscheduled learning rate that alone was capping accuracy at chance-to-single-digits**. Fixing only the schedule, with zero architecture changes, recovered 45–65% accuracy from a config that had been stuck at 6.3%. The attention baseline has never been given the same optimizer treatment. **No number in this repo currently supports a claim that AAC-diff is better or worse than attention.** |

If you're looking for "does AAC work," the honest answer right now is:
**unknown, pending a fair re-run.** The next required step (not yet
done) is re-testing every config — including the attention baseline —
under a properly explored learning-rate schedule, to confirmed
convergence, before any comparative claim is meaningful again.

## Repo structure

```
AAC.txt                    original architecture specification
AAC-corrections.md         math errors found in the spec, and their fixes

aac_prototype/             Phase 1: non-differentiable NumPy prototype
  aac/                     fast_state, memory, trace, controller, mlp, model
  train.py                 trains on a hand-aligned synthetic recall task
  mechanism_checks.py      isolated routing/merge/eviction stress tests
  scaling_sweep.py         routing latency scaling, generalization sweep

  aac/autodiff.py          hand-rolled, gradient-checked reverse-mode autodiff
  aac/models_diff.py       RNN baseline, attention baseline, differentiable
                           AAC-diff (all matched-budget, all trained end-to-end)
  task_mqar.py              harder shared-vocabulary induction-head recall task
  train_compare.py          Phase 2 training/comparison driver
  test_autodiff.py          numerical gradient checks for every autodiff op

PHASE2_RESULTS.md           Phase 2 write-up
DELTA_RULE_RESULTS.md       delta-rule variant results
PHASE3_GATE_RESULTS.md      Phase 3 write-up (supersedes Phase 2's headline claim)
```

No PyTorch/JAX is used anywhere in the differentiable work — there was
no network access available to install one during development, so
`aac/autodiff.py` is a small hand-written, gradient-checked reverse-mode
autodiff engine. This is a real limitation worth knowing about if you're
evaluating the engineering here: it's correct (verified to ~1e-10
against finite differences, including the exact multi-step
recurrence-plus-memory pattern the models use), but it's slower and less
battle-tested than a real framework, and porting to one is a natural
next step.

## Running it

```bash
pip install numpy   # the only dependency, anywhere in this repo

# Phase 1: non-differentiable prototype
cd aac_prototype
python3 train.py
python3 mechanism_checks.py
python3 scaling_sweep.py

# Phase 2/3: differentiable comparison
python3 test_autodiff.py     # verify the autodiff engine before trusting anything else
python3 train_compare.py     # RNN vs. attention vs. AAC-diff, matched budget
```

## What's genuinely established so far

- The discrete write/promote/evict/merge/hierarchical-routing mechanisms
  work as specified, in isolation, when given hand-aligned inputs
  (Phase 1).
- A differentiable relaxation of the memory is trainable end-to-end via
  real backprop (verified via gradient checking, not assumed).
- The read/retrieval pathway is *not* independently broken — it learns
  fine given correct writes (Phase 3, Finding 1, correcting an earlier
  Phase 2 misdiagnosis).
- A fixed, fast exponential decay silently caps the usable memory
  horizon regardless of write quality — this is a real, fixable
  bottleneck, confirmed stable across every later check (Phase 3,
  Finding 2).
- Optimizer hygiene (learning-rate scheduling) has a larger effect on
  measured accuracy than every architectural variable tested in this
  repo so far, combined. This was not anticipated going in, and is
  arguably the most important finding to date — not because it's about
  AAC specifically, but because it invalidates the basis for comparing
  anything else until it's controlled for.

## What's genuinely still open

- Whether AAC-diff can match or beat attention once both are trained
  under equivalent, converged optimization — **not yet tested**.
- Whether gate supervision helps net of a fixed optimizer, and by how
  much — current estimate (+5-9pp, 2 seeds, 2 unconverged schedules) is
  a lower-confidence placeholder, not a settled number.
- Everything about the discrete mechanisms (promotion, eviction, merge,
  hierarchical routing, the controller's adaptive-reasoning budget) in
  the *differentiable* setting — none of it has been integrated with the
  trainable pathway yet. Phase 2/3 test a single fixed-size associative
  matrix as a stand-in for the entire persistent-memory subsystem.
- Any test beyond toy scale (currently `n_vocab=64`, hidden dim 32,
  single layer, synthetic tasks only).

## Contributing / next steps

In priority order, per Phase 3's own conclusion:

1. Re-run every existing configuration — RNN, attention, and AAC-diff,
   aux and no-aux — under a properly explored learning-rate schedule
   space, to confirmed convergence, with the *same* optimizer treatment
   applied to all of them. Nothing else in this list matters until this
   is done, because it's the thing that would make any other comparison
   trustworthy again.
2. Only after (1): investigate an unsupervised (non-oracle-label) proxy
   for write importance, since the current gate-supervision results
   depend on ground-truth labels not available in a real deployment.
3. Only after (1) and (2): consider read-side sharpening (e.g. a
   softmax/Hopfield-style retrieval instead of the current unnormalized
   linear dot-product read) as a separate, isolated experiment — not
   bundled with (2), so any effect can be attributed correctly.
4. Longer-term: integrate the discrete mechanisms from the Phase 1
   prototype (promotion, eviction, merge, hierarchical routing, adaptive
   reasoning) into the differentiable pathway, once the foundation above
   is stable enough to build on without re-litigating basic optimizer
   hygiene every time a new number looks surprising.

If you're adding a new experiment: gradient-check any new autodiff op
before trusting results built on it, run more than one seed before
reporting an accuracy number, and check for convergence before treating
any number as final. Every phase in this repo that skipped one of those
three things had to walk something back later.

## License

*Apache 2.0*
