# Phase 3: Isolating and Partially Fixing the Write-Gate Failure

Phase 2 (`PHASE2_RESULTS.md`) established that AAC-diff loses badly to
softmax attention on MQAR and traced this to the write gate never
learning selectivity (key-position gate values roughly equal to
filler-position values, ratio ~0.8-1.0x across every update rule tested
in `DELTA_RULE_RESULTS.md`). That work left the *cause* open: is it the
gate, the read/retrieval mechanism, the fixed exponential decay, or some
mix, and is it fixable at all?

This phase isolates each of those variables independently using
controlled ablations of `aac/models_diff.py`'s `AACDiffModel`, and finds:
the read mechanism is fine, the decay rate is a real and independently
fixable bottleneck, the gate-selectivity failure is real and is a second
dominant bottleneck, three unsupervised fixes for it all fail, and a
supervised fix does teach real, low-variance selectivity even under fast
decay. Selectivity alone doesn't help unless decay is also loosened.

A rigorous post-hoc robustness check (five seeds, a hyperparameter
sweep, extended training to convergence) found the resulting *accuracy*
considerably less reliable than the selectivity that produces it -- and
chasing down *why* overturned the phase's central claim rather than just
refining it. The instability traced to a constant, unscheduled learning
rate used throughout this entire investigation (and Phase 2 before it).
Fixing the schedule with **zero gate supervision** recovered 45-65%
held-out accuracy from a no-aux baseline that had been stuck at 6.3%
under constant LR -- a larger effect than anything gate-related tested
anywhere in this phase. Auxiliary gate supervision still contributes a
real, separable effect on top of that (+5-9 percentage points at a
matched schedule, confirmed at two seeds) -- genuine, but secondary, not
"the dominant bottleneck" as Finding 3 originally claimed. None of these
numbers are confirmed converged (three of four configurations are still
rising at 9000 episodes), and the specific LR-decay numbers turned out
to be sensitive to the schedule's exact duration, not just its presence.
See "Isolating the LR-decay effect from gate supervision" for the full
accounting, including a previously-unnoticed, now-fixed reproducibility
gap in the autodiff engine found while chasing this down.

All numbers below: `n_pairs=6`, `n_vocab=64` (chance = 1.6%), `d_emb=d_h=32`,
Adam `lr=1e-2`, batch=8, MQAR task (`task_mqar.py`). Episode counts vary by
experiment (2000-4000) and are noted per section; this is a smaller budget
than Phase 2's 3000, chosen for iteration speed during this investigation,
so absolute numbers below are directionally reliable but a final write-up
should re-run the winning configurations at matched 3000+-episode budget
for a clean head-to-head against Phase 2's baselines.

## Method: three new `AACDiffModel` gate modes

- `gate_mode="oracle"`: hard 1/0 write exactly at the true key→value
  binding position, **no decay applied at all** (`M += outer(k,v)`,
  unconditionally). Not a controlled variable in isolation -- see below.
- `gate_mode="oracle_decay"`: same hard, correctly-timed 1/0 write, but
  **with** the same `decay` multiplier every other mode uses
  (`M = decay*M + mask*outer(k,v)`). This isolates "correct write timing"
  from "no forgetting at all."
- `gate_mode="learned"` (existing): `g = sigmoid(Wg @ h + bg)`, with an
  optional warm-start bias `bg`, an optional curriculum blend against the
  oracle mask (`gate_mode="curriculum"`, ramped `alpha`), and an optional
  auxiliary supervised loss (`aux_gate_weight`, `aux_pos_weight`) --
  weighted binary cross-entropy between the gate's sigmoid output and the
  true oracle write label at every step, added directly into the same
  backward pass as the task loss. Implemented as a new `ad.bce_loss` op in
  `aac/autodiff.py`, gradient-checked to ~1e-11 against finite differences.

## Finding 1 -- the read/retrieval mechanism is not broken

Phase 2's "perfect-write test" (`train_compare.py::perfect_write_test`)
found near-chance accuracy (0.7-3.3%) even when `M` was hand-constructed
from ground-truth pairs, and concluded the read mechanism (`Wq`, `Wk`,
`Wv`, linear dot-product retrieval, the readout combiner) might be
*independently* broken -- a "dual bottleneck."

That conclusion doesn't hold up. That test froze a perfect `M` on top of
`Wq/Wk/Wv/Wcomb/Wout` weights that had been trained under the normal
*broken* regime, where the gate never fires and those weights never see a
useful gradient signal. Training the full model **jointly**, from
scratch, with the gate replaced by the oracle mask, gets:

| Model (2000 ep) | Held-out acc | Gap 20-40 | Gap 60-90 | Gap 150-200 | Gap 300-400 |
|---|---:|---:|---:|---:|---:|
| RNN-only (floor) | 1.6% | -- | -- | -- | -- |
| AAC-diff, learned gate | 1.1% | 1.7% | 1.2% | 1.8% | 1.5% |
| **AAC-diff, oracle gate (no decay)** | **98.8%** | 99.0% | 97.5% | 98.8% | 98.0% |

The read pathway learns fine and generalizes to 10x the training gap
length, given correct writes. The earlier "dual bottleneck" conclusion
was an artifact of testing an undertrained read pathway, not a
structural defect in linear dot-product retrieval. **Correction to
`PHASE2_RESULTS.md`'s perfect-write-test interpretation.**

## Finding 2 -- the fixed decay rate is a second, independent, fixable bottleneck

The oracle-gate result above is confounded: `gate_mode="oracle"` also
removes decay entirely, so it isn't testing "correct write timing" in
isolation. Controlling for this with `oracle_decay` (correct timing,
decay reapplied):

| Write timing | decay | Held-out (gap 20-40) | Gap 60-90 | Gap 150-200 | Gap 300-400 |
|---|---:|---:|---:|---:|---:|
| oracle | 0.95 | 77.2% | 60.0% | 1.0% | 2.8% |
| oracle | 0.99 | 99.0% | 99.5% | 98.0% | 35.7% |
| **oracle** | **0.999** | **99.0%** | **99.5%** | **98.7%** | **99.7%** |

With perfect write timing, `decay=0.95` alone caps the usable memory
horizon at roughly `1/(1-0.95) ~ 20` steps (`0.95^150 ~ 3e-4`) -- by
gap 150-200 accuracy collapses to chance even though every write was
timed correctly. `decay=0.999` removes this ceiling almost entirely and
holds near-perfect accuracy out to 10x the training gap length. **This
confirms the fixed geometric decay is a real, severable bottleneck, and
it is fully fixable independent of gate quality.**

## Finding 3 -- decay alone does not fix, and does not even move, gate selectivity

Same decay sweep, but with the gate `learned` instead of `oracle`:

| decay | Held-out | Gate value at key positions | Gate value at filler | Ratio |
|---|---:|---:|---:|---:|
| 0.95 | 1.6% | 0.218 | 0.218 | 1.00x |
| 0.99 | 4.8% | 0.295 | 0.283 | 1.04x |
| 0.999 | 6.3% | 0.381 | 0.383 | 0.99x |

Absolute gate magnitude rises with slower decay (0.22 -> 0.38, presumably
because a weaker, less-decayed signal manages to reduce loss a little via
uniformly-more-open writing) but **the key/filler ratio stays pinned at
~1.0x across all three settings** -- the gate never learns to
discriminate positions at all, regardless of how forgiving the rest of
the system is. Two unsupervised interventions were also tried and both
failed to move the ratio in a useful direction:

| Intervention (2000 ep, decay=0.95) | Held-out | Gate ratio |
|---|---:|---:|
| Warm-start bias `bg=+2.0` (initial gate ~0.88) | 1.3% | 1.05x (converged to ~uniformly-open, 0.60-0.63 both classes) |
| Warm-start bias `bg=+4.0` (initial gate ~0.98) | 1.7% | 1.02x (same failure mode) |
| Curriculum: oracle write -> learned write, alpha ramped 0->1 over 60% of training | 1.5% | 1.76x, but magnitude collapses to ~0.02-0.04 (same as the unsupervised baseline) |

Warm-starting the gate open just finds a new local optimum -- "write
everything, uniformly" -- instead of "write nothing"; neither is
selective. The curriculum run is the most informative negative result:
even while `alpha` is still mostly weighted toward the oracle mask
(e.g. `alpha=0.21` at episode 250, so 79% of the actual write is still
oracle-timed), training accuracy stays at chance (~1.5-2.4%) throughout
-- far below what the corresponding oracle-timed run achieves at the
same episode count (e.g. `oracle_decay`, decay=0.95 hits 40%+ train
accuracy by episode 1000). The scaffolding doesn't transfer any usable
signal to the learned gate even while it's doing most of the work,
which is consistent with "write selectivity is a hard local-optimum
problem that isn't solved by making it directionally easier."

## Finding 4 -- direct supervision on the gate is the first thing that works

Added a weighted binary cross-entropy auxiliary loss directly on the
gate's sigmoid output against the true oracle write label
(`aux_gate_weight`, `aux_pos_weight` to counter the severe class
imbalance between rare key-write positions and abundant filler
positions), summed into the same backward pass as the task loss
(new `ad.bce_loss` op, gradient-checked to 1e-11).

**Full sweep, 2000 episodes, confirmed on local hardware (`run_aux_gate.py`,
SEED=0)** -- superseding the earlier non-converged sandbox estimate:

| Config | Held-out (gap 20-40) | Gate ratio (key/filler) | Gate value, key | Gate value, filler |
|---|---:|---:|---:|---:|
| learned, no aux (baseline) | 1.8% | 0.95x | 0.123 | 0.130 |
| learned + aux(w=1, posw=5), decay=0.95 | 0.7% | **1.64x** | 0.394 | 0.241 |
| learned + aux(w=1, posw=10), decay=0.95 | 1.8% | **1.24x** | 0.518 | 0.417 |
| learned + aux(w=3, posw=5), decay=0.95 | 1.7% | **1.59x** | 0.385 | 0.242 |
| **learned + aux(w=1, posw=5), decay=0.999** | **26.4%** | **1.77x** | 0.391 | 0.221 |

Generalization for the decay=0.999+aux config (the only one worth sweeping,
since the others never leave chance):

| Gap 20-40 | Gap 60-90 | Gap 150-200 | Gap 300-400 |
|---:|---:|---:|---:|
| 25.5% | 23.5% | 16.0% | 8.5% |

This degrades *gracefully* -- roughly halving every ~150-200 extra steps
of gap -- rather than collapsing sharply to chance the way every other
learned-gate configuration in this phase does. That graceful-degradation
shape is qualitatively closer to how the Phase 2 attention baseline
behaves (57.2% -> 17.7% over a similar gap range) than to AAC-diff's
previous cliff-edge failures, even though the absolute numbers are still
well below attention's.

### The new finding: partial selectivity alone is not enough -- it needs decay to also be loosened

The three `decay=0.95` aux rows above are the interesting new result.
**The auxiliary loss unambiguously produces real gate selectivity even
under fast decay** -- ratios of 1.24x-1.64x, nothing like the dead-flat
~1.0x every unsupervised intervention produced in Finding 3. But **none
of it shows up in accuracy** -- all three stay pinned at chance (0.7-1.8%),
including at the training-distribution gap length, exactly where
oracle-timed hard writes under the same `decay=0.95` reached 77.2%
(Finding 2).

So gate selectivity and decay are not simply two independent knobs that
each contribute additively -- there's a **threshold effect** between
them. A soft, partially-selective gate (magnitude ~0.2-0.5, ratio
~1.2-1.6x) is not the same as the oracle's hard 0/1 write, and under
`decay=0.95` that gap matters enormously: the small amount of "leakage"
onto filler positions, and the fact that even correct writes aren't at
full strength, is apparently enough for the aggressive per-step decay to
wash the signal out before it reaches a query. Only once decay is also
relaxed (`0.999`) does that same partial selectivity translate into a
15x accuracy jump (1.8% -> 26.4%). **Fixing gate selectivity only pays
off once decay is loose enough for a moderately-selective, non-binary
gate to still matter by the time a query arrives.**

**Everything above this point was written after a single seed, a single
hyperparameter setting, and a training run that had not been checked for
convergence. The section below stress-tests all three of those choices,
and the headline number does not fully survive.**

## Robustness check

Before treating the 26.4% result as a finding rather than a promising
single-seed data point, four things needed checking: whether the engine
fixes from this phase could have silently altered any *prior* phase's
numbers, whether the headline config replicates across seeds, whether it
was actually the best available setting, and whether it had converged.
All four were run to completion; two came back clean, two did not.

### 1. Did the autodiff fixes affect Phase 2 / `DELTA_RULE_RESULTS.md`? -- No, confirmed both analytically and empirically.

Measured actual recursion depth (not estimated) needed for `.backward()`
on real episodes, using the traversal's own max stack size as a direct
proxy for what the old recursive version would have needed:

| Config | Max token length | Max depth needed | vs. Python's limit (1000) |
|---|---:|---:|---|
| Training gap (20-40), RNNBaseline | 91 | 361 | safely under |
| Training gap (20-40), AttentionBaseline | 86 | 19 | safely under (attention has no recurrent chain -- depth is ~constant regardless of sequence length, structurally immune to this class of bug) |
| Training gap (20-40), AACDiffModel | 91 | 350 | safely under |
| Eval gap (300-400), RNNBaseline | 447 | 1766 | **over the limit** |
| Eval gap (300-400), AACDiffModel | 451 | 1801 | **over the limit** |

The eval-gap numbers exceed the limit, but this is not a problem for
Phase 2's actual numbers: **the pre-fix code never called `.backward()`
in eval mode at all** (`if train: total_loss.backward()`, with no `else`
branch), so the recursive traversal never ran during any of Phase 2's
gap-length sweep -- those long graphs were computed forward (no
recursion involved there) and simply discarded, never crashing and never
silently computing anything wrong. Training always used the short
default gap, safely under the limit. So there is no code path in Phase 2
that could have been affected by either engine fix.

Confirmed this isn't just an argument by re-running Phase 2's exact
`train_compare.py` procedure (3000 episodes, all three models, full gap
sweep) on the current fixed engine:

| Model | Held-out (this run) | Held-out (published) | Gap 300-400 (this run) | Gap 300-400 (published) |
|---|---:|---:|---:|---:|
| RNN-only | 1.0% | 1.4% | -- | 1.5% |
| Attention | **57.2%** | **57.2%** | **17.7%** | **17.7%** |
| AAC-diff (Hebbian) | 1.1% | 1.4% | 2.1% | 3.1% |

Attention matches to the reported precision (expected -- its graph never
approaches the recursion depth where old and new traversal could ever
have differed, so this is essentially a bit-identical rerun). RNN-only
and AAC-diff differ by ~0.3-1.0pp, consistent with ordinary single-seed
sampling noise at chance-level accuracy, not a computational discrepancy.
**Phase 2 and `DELTA_RULE_RESULTS.md` are confirmed unaffected.**

### 2. Multi-seed the winning config -- the headline number does not replicate reliably.

`decay=0.999`, `aux_gate_weight=1`, `aux_pos_weight=5`, 2000 episodes,
5 seeds, each with a **fresh** `MQARTask` (not the sequential task-reuse
pattern `run_aux_gate.py` used across its 5-config sweep, which meant
the original 26.4% run never actually trained on a clean seed=0 episode
stream to begin with -- it trained on whatever came after ~8000 episodes
of prior RNG consumption from the four earlier configs in that script.
That's not a bug, but it does mean 26.4% was never a clean, reproducible
single-seed number in the first place):

| Seed | Held-out acc | Gate ratio |
|---:|---:|---:|
| 0 | 12.8% | 1.39x |
| 1 | 25.1% | 1.73x |
| 2 | 22.3% | 1.47x |
| 3 | **33.1%** | 1.74x |
| 4 | **1.8%** | 1.55x |

**Held-out accuracy: mean 19.0%, std 10.8%, range 1.8%-33.1%.**
**Gate ratio: mean 1.58x, std 0.14x, range 1.39x-1.74x.**

The spread in accuracy is large relative to the 26.4% point estimate --
larger than the point estimate itself, in fact (std 10.8pp on a mean of
19.0pp). One seed (4) is statistically indistinguishable from the
`decay=0.95` baseline's chance-level failure. **26.4% should not be
reported as "the" result of this configuration; it is one draw from a
wide distribution.**

The gate ratio, in sharp contrast, is tight (std 0.14x) and clears 1.0x
in every single seed, including seed 4 -- which nonetheless produced
only 1.8% accuracy. **Gate selectivity reliably emerges under this
configuration. Task accuracy does not reliably follow from it.** This
decoupling recurs throughout the checks below and is now the single most
important open question this phase leaves unresolved (see Interpretation).

### 3. Sweep `aux_pos_weight` x `aux_gate_weight` at `decay=0.999` -- real headroom over the headline config, at this budget.

Single seed (0), fresh task per cell, 2000 episodes, full 4x3 grid:

| gate_w | pos_w | Held-out | Gate ratio |
|---:|---:|---:|---:|
| 3.0 | 2.0 | **23.4%** | 3.04x |
| 1.0 | 2.0 | 22.5% | 2.43x |
| 3.0 | 10.0 | 21.7% | 1.38x |
| 3.0 | 5.0 | 21.5% | 1.51x |
| 0.5 | 2.0 | 19.2% | 2.19x |
| **1.0** | **5.0** | **16.3%** (headline) | 1.73x |
| 1.0 | 10.0 | 15.8% | 1.28x |
| 3.0 | 20.0 | 12.2% | 1.14x |
| 0.5 | 10.0 | 11.2% | 1.19x |
| 1.0 | 20.0 | 10.2% | 1.14x |
| 0.5 | 20.0 | 7.2% | 1.14x |
| 0.5 | 5.0 | **2.5%** | 1.58x |

The headline config is 6th of 12 at this seed/budget -- not the best.
`pos_weight=2` dominates the top of the grid (3 of the top 5 rows). The
bottom row is another clean example of the ratio/accuracy decoupling:
`gate_w=0.5, pos_w=5` gets a perfectly respectable 1.58x ratio and 2.5%
accuracy -- worse than several configs with *lower* ratios.

### 4. Extended training to 6000 episodes -- the "better" config from the sweep turns out to be unstable, not just less-converged.

Took the sweep's best cell (`gate_w=3.0, pos_w=2.0`) and the headline
config (`gate_w=1.0, pos_w=5.0`), ran both to 6000 episodes with a
held-out checkpoint (150 episodes, fixed eval seed) every 1000 episodes:

| Episode | Headline (gw=1, pw=5) | "Best-in-grid" (gw=3, pw=2) |
|---:|---:|---:|
| 1000 | 1.2% | 0.9% |
| 2000 | 11.4% | 24.4% |
| 3000 | 16.3% | **28.4% (peak)** |
| 4000 | 18.7% | 27.9% |
| 5000 | 19.0% | 22.9% |
| 6000 | **18.2%** | **19.4%** |
| Gate ratio @ 6000 | 1.61x | 2.07x |

Two different dynamics, not one delayed-vs-fast version of the same
curve: the headline config rises and **stabilizes** around 18-19%. The
sweep's "best" config rises faster, **peaks near episode 3000 at 28.4%,
and then genuinely declines** to 19.4% -- it isn't undertrained, it's
unstable, and the 2000-episode sweep in check 3 caught it near a
transient peak rather than a converged value. By episode 6000, the two
configs that looked 7 points apart at the sweep's budget land within
1.2 points of each other. **The "meaningfully more headroom" finding
from check 3 does not survive longer training** -- or at least, not in
the form it first appeared in.

Neither curve is cleanly monotonic-and-still-rising the way the original
(single, non-converged) 4000-episode sandbox run of the headline config
suggested. The most likely explanation for the peak-then-decline pattern
is a plain optimization one that this phase hasn't controlled for: every
run in this entire investigation uses a constant Adam learning rate
(1e-2) with no decay or schedule. That's a very ordinary source of
late-training oscillation once a model has found a reasonable region of
parameter space, and it hasn't been ruled out here.

## Resolving the gate-ratio/accuracy decoupling

The robustness check above left one open question: gate selectivity
(ratio) is stable across seeds and hyperparameters while task accuracy
is not -- seed 4 of the multi-seed run and the `gate_w=0.5, pos_w=5`
sweep cell both reached a respectable ratio (~1.5-1.6x) paired with
near-chance accuracy. Three candidate explanations were proposed, to be
tested cheapest-first with an explicit stopping rule. Two were needed;
the second one resolved it completely.

### A prerequisite finding: "same seed" reruns in this engine are not bit-identical

While re-extracting gate magnitudes for check 1 (below), rerunning the
sweep's 12 configs with identical seeds and identical code did **not**
reproduce the original sweep numbers exactly (e.g. `gate_w=0.5,
pos_w=10`: 11.2%/1.19x originally, 19.3%/1.48x on rerun). Root cause:
`Tensor._prev` is a plain `set(_children)`, and `Tensor` defines no
custom `__hash__`/`__eq__`, so it falls back to Python's default
identity-based hash. Iteration order over a set of objects hashed by
identity depends on their memory addresses, which are not fixed by any
RNG seed and can differ between separate process invocations. Floating-
point addition is not exactly associative, so `_backward` closures that
accumulate gradients from multiple parents into a shared child
(`a.grad += out.grad`) can sum in a different order run-to-run, and that
bit-level noise compounds over ~2000 episodes of nonlinear Adam
optimization into visibly different trained models -- even with every
explicit RNG seed in the codebase held fixed.

This means **every "same-seed rerun" in this entire phase carried this
hidden extra source of variation**, layered on top of whatever seed
difference was deliberately being tested. It doesn't invalidate the
multi-seed or sweep findings (those were already explicitly about
run-to-run variance, and this just means the true variance floor is
slightly higher than "seed" alone would suggest), but it does mean
none of the specific numbers in this document should be treated as
exactly reproducible from the stated seed alone. Noted here and in the
Methods note section below; not fixed in this pass (a content-based
tie-break, e.g. an incrementing creation-order counter used as a sort
key instead of relying on set iteration, would remove it if exact
reproducibility becomes a priority).

### Step 1: gate-magnitude correlation -- ruled out, and not just "no correlation"

Combined the 5 multi-seed runs (already logged with per-run key/filler
gate magnitude) with the 12 sweep cells (re-extracted with magnitude
logging this time, using the same deterministic configuration --
subject to the caveat directly above) for 17 total data points:

| | corr with held-out accuracy |
|---|---:|
| Gate ratio (key/filler) | **+0.51** (moderate, right-signed) |
| Mean absolute gate value at KEY positions | **-0.33** (weak, *wrong*-signed) |
| Mean absolute gate value at FILLER positions | -0.47 (moderate, negative) |
| (for reference) key-gate magnitude vs. ratio | -0.74 (strong negative -- pushing magnitude up tends to cost ratio, not build on it) |

The hypothesis specifically predicted that the two low-accuracy examples
would cluster at *low* key-gate magnitude. They don't: seed 4's key-gate
magnitude (0.331) and `gate_w=0.5,pos_w=5`'s (0.364) both sit in the
*middle* of the full 0.155-0.667 range across all 17 points. The single
most extreme low-magnitude point in the entire dataset
(`gate_w=0.5,pos_w=2`, key-gate = 0.155) gets one of the *better*
accuracies in the sweep (18.9%) -- the direct opposite of the predicted
relationship. **Ruled out explicitly, not just unconfirmed.** Per the
stopping rule, proceeded to step 2.

### Step 2: LR-schedule / optimization-instability check -- confirmed, decisively, and bigger than expected

Reran four configurations with a linear learning-rate decay
(`1e-2 -> 1e-3` over 6000 episodes, replacing the constant `1e-2` used
everywhere else in this phase) and nothing else changed: the same two
configs from Robustness-check item 4, plus -- to test the *original*
decoupling directly rather than inferring from different configs --
both configurations specifically named in the open question.

| Config | Constant LR, final (ep6000 or ep2000) | With LR decay, ep6000 | 
|---|---:|---:|
| Headline (`gw=1, pw=5`) | 18.2% (stable plateau) | **43.9%**, still rising |
| "Best-in-grid" (`gw=3, pw=2`) | 19.4% (after peaking at 28.4%, then declining) | **51.9%**, monotonic, no decline |
| **Seed 4** (originally 1.8% @ 2000 ep) | 1.8% | **76.7%** |
| **`gw=0.5, pw=5`** (originally 2.5% @ 2000 ep) | 2.5% | **78.9%** |

Every single config improves dramatically. The peak-then-decline pattern
in the "best-in-grid" config is completely gone under LR decay, replaced
by sustained improvement (23.9% -> 41.4% -> 48.6% -> 51.9%, no downturn).
And critically, **both of the specific cases that originally defined the
"fine ratio, near-chance accuracy" puzzle turn out to reach the best
results in this entire investigation** -- 76.7% and 78.9% respectively,
comfortably exceeding even the Phase 2 attention baseline's 57.2%, using
the exact same seed, data ordering, and weight initialization that
produced near-chance accuracy under a constant learning rate.

**This is a clean, complete resolution, not a partial one.** The
ratio/accuracy decoupling was never evidence of some deeper mechanism
(a separately seed-sensitive read pathway, or gate quality being
insufficient by some other measure) -- it was ordinary optimization
instability from training at a constant, too-high learning rate for
6000 (or even 2000) episodes with no decay. The gate's ratio apparently
stabilizes into a reasonable value largely independent of whether the
rest of the network has converged; accuracy depends on the rest of the
network actually converging, which a constant high LR frequently
prevents, sometimes for the full training run. **Step 3 (the read-
pathway re-initialization test) is unnecessary** -- the stopping rule in
the original request is satisfied.

### An urgent follow-up this result demands, not yet tested

The size of this effect raises a question this document cannot yet
answer: **does LR decay alone -- with no auxiliary gate supervision at
all -- also rescue the plain `decay=0.999, no aux` configuration** (ratio
0.99x, no selectivity whatsoever, 6.3% held-out under constant LR;
Finding 3)? If it does, that would mean optimization scheduling, not
auxiliary gate supervision, was the dominant lever in this phase all
along, and Findings 4/5's causal story ("selectivity requires
supervision; selectivity requires loose decay to matter") would need to
be re-examined for how much of the accuracy gain they actually
contributed once optimization is no longer artificially crippled. This
is the single most important next experiment before any number in this
phase goes into a paper.

**This follow-up has now been run.** See the next section.

## Isolating the LR-decay effect from gate supervision

Three things needed to happen before the LR-decay numbers went anywhere
near a conclusion: fix the non-deterministic tie-break found above (so
the comparison isn't contaminated by the same noise that already
produced one contradictory rerun), run the no-aux control under
identical conditions, and check convergence properly this time instead
of reading a snapshot. All three are done. The outcome doesn't match
either of the two clean hypotheses that were on the table -- it's a
real mixture of both, and that's reported plainly below rather than
forced into one bucket.

### Fix 1: the non-deterministic tie-break is resolved

`Tensor._prev` was a plain `set(_children)`; since `Tensor` defines no
custom `__hash__`/`__eq__`, that fell back to Python's identity-based
default, whose iteration order depends on memory addresses rather than
anything seeded. Fixed by giving every `Tensor` an incrementing
creation-order id at construction and replacing the set with a list of
unique children sorted by that id -- deterministic regardless of memory
layout, and a safe drop-in since `_prev` is never used for anything but
iteration anywhere in the codebase (confirmed by grep before making the
change).

Verified two ways: the full gradient-check suite still passes to
~1e-10 (confirming no computation changed, only traversal order), and
the same config/seed run twice in **separate process invocations** now
produces bit-identical results to 15 decimal places (`Wg` weight sum,
`Wk` weight sum, and final loss all matched exactly) -- previously,
identical seeds gave different numbers from run to run (the
`gate_w=0.5,pos_w=10` discrepancy noted above was an instance of this).
This fix is now in place for everything that follows.

### Step 1 result: the no-aux control, at the original 6000-episode schedule -- mixed, not clean

Reran `gate_mode="learned"`, `decay=0.999`, `aux_gate_weight=0` (Finding
3's exact failing config) with the same linear LR decay (`1e-2 -> 1e-3`
over 6000 episodes) as the four aux-supervised reruns, at the same two
seeds already tested (0 and 4):

| Seed | No-aux, LR decay (6000 ep) | Aux, LR decay (6000 ep) | Gap |
|---:|---:|---:|---:|
| 0 | **48.1%** | 43.9% | no-aux **ahead** by 4.2pp |
| 4 | 64.6% (peak 66.7% @ ep5000) | **76.7%** | aux ahead by ~10-12pp |

Neither of the two pre-specified outcomes materialized cleanly. At seed
0, the no-aux config *beats* the aux-supervised one. At seed 4, aux
clearly wins by a real margin. And in both cases, the no-aux config
learns **real gate selectivity with zero supervision** -- ratio 1.29x
(seed 0) and 1.61x (seed 4) -- something Finding 3 explicitly reported
never happening under constant LR (ratio stuck at ~1.0x across every
unsupervised intervention tried there). That specific claim in Finding 3
does not survive this check; see Interpretation below.

### An unplanned discovery while checking convergence: results are sensitive to the LR schedule's *duration*, not just its presence

Extending training to check convergence (item 3) required deciding how
to extend the schedule. Rescaling the linear decay to span the full
9000 episodes (rather than reusing the 6000-episode schedule and holding
flat at `1e-3` for the remainder) turned out to matter enormously: at
episode 6000 *within the 9000-episode schedule*, the LR has only reached
~4e-3 (partway through its decay), not the ~1e-3 it would be under the
original 6000-episode schedule. The result is a substantially different
trajectory, not a continuation of the earlier one -- e.g. `aux, seed 4`
reaches 76.7% at episode 6000 under the 6000-episode schedule, but only
47.2% at that same episode count under the 9000-episode schedule. **The
specific numbers in the "urgent follow-up" section above and in
Robustness-check item 4 are tied to that one specific schedule
(1e-2->1e-3 linear over exactly 6000 episodes), not a general property
of "LR decay helps."** This is now flagged as its own caveat, separate
from the gate-supervision question.

Given that, the only methodologically clean way to compare aux vs.
no-aux is on a **matched** schedule. All four configs below use the same
9000-episode linear decay:

| Seed | Aux (9000-ep schedule) | No-aux (9000-ep schedule) | Gap |
|---:|---:|---:|---:|
| 0 | 57.7% | 52.6% | +5.1pp |
| 4 | 53.8% | 45.0% | +8.8pp |

With a matched schedule, the picture is more consistent than the
6000-episode comparison above: aux beats no-aux at both seeds, by a
real but modest 5-9 percentage points -- nowhere near the ~40-75
percentage-point jump that LR decay itself produces over the constant-LR
baseline (~1-6%) in every single configuration, aux or not.

### Step 3 result: convergence check -- three of four configs are still rising at 9000 episodes

Checked the stopping rule (three consecutive 1000-episode checkpoints
within 2pp of each other) against the last three checkpoints of each
9000-episode run:

| Config | ep7000 | ep8000 | ep9000 | Deltas | Converged? |
|---|---:|---:|---:|---:|---:|
| Aux, seed 0 | 51.7% | 54.8% | 57.7% | 3.1pp, 2.9pp | **No** |
| Aux, seed 4 | 48.6% | 53.0% | 53.8% | 4.4pp, 0.8pp | **No** |
| No-aux, seed 0 | 43.9% | 49.2% | 52.6% | 5.3pp, 3.4pp | **No** |
| No-aux, seed 4 | 42.0% | 43.1% | 45.0% | 1.1pp, 1.9pp | **Yes** (barely) |

Only one of four configurations meets the stopping criterion, and even
that one only barely (both deltas just under the 2pp threshold, from
two data points, not a settled plateau). **None of the numbers in this
section -- 45-58% across the board -- should be read as converged,
asymptotic values.** All four are still climbing at episode 9000, most
of them substantially. This phase already misread one non-converged
curve as a stable result once (Robustness-check item 3/4); the honest
statement here is that the true long-run accuracy of every configuration
in this document, aux or no-aux, LR-decayed or not, remains unknown, and
determining it needs longer runs than this investigation has budgeted
for so far.

### Net effect: a hybrid answer, not either of the two proposed ones

- **The dominant lever is unambiguously the learning-rate schedule, not
  gate supervision.** No-aux configurations went from ~1-6% (constant
  LR, Finding 3) to 45-53% (matched 9000-episode LR decay) with *zero*
  changes to the gate mechanism -- a 40-50 percentage-point jump from
  optimization scheduling alone. This is the single largest effect size
  found anywhere in this entire phase, dwarfing every gate-related
  intervention tried (oracle timing, decay sweeps, warm-start,
  curriculum, and auxiliary supervision itself).
- **Auxiliary gate supervision still does real, separable work on top of
  that** -- a consistent (if much smaller) 5-9 percentage-point
  advantage over no-aux at a matched schedule, in both seeds tested.
  This is the actual, honest effect size of gate supervision that the
  document did not previously have a number for.
- **Finding 3's strong claim does not survive.** "Gate selectivity...is
  not a soft optimization difficulty that yields to easier initialization
  or curriculum scaffolding" is contradicted directly: the plain,
  completely unsupervised gate learns real selectivity (ratio 1.29-1.61x)
  once the optimizer is not artificially crippled by a constant,
  unscheduled learning rate. It *was* a soft optimization difficulty, at
  least in large part. The weaker, revised claim -- supervision adds a
  further, smaller, real selectivity/accuracy boost on top of a properly
  optimized baseline -- does survive, at the ~5-9pp level shown above.
- **None of this is converged**, and the schedule-duration sensitivity
  found by accident while checking convergence means even the
  now-superseded 6000-episode numbers (76.7%, 78.9%) were artifacts of
  one specific, arbitrarily-chosen schedule length, not a general
  "AAC-diff reaches ~77%" result. The true long-run numbers, and how
  they compare to Phase 2's attention baseline (also never tested with
  LR decay -- see the earlier caveat), remain open questions.


## Summary table

|---|---:|---:|---|
| RNN-only (no memory) | 1.6% | -- | floor |
| Learned gate, decay=0.95 (Phase 2 config) | 1.1-1.8% | 0.95-1.00x | baseline failure |
| Oracle write timing, no decay | 98.8% | -- (n/a) | read mechanism is fine |
| Oracle write timing, decay=0.95 | 77.2% | -- (n/a) | write-timing alone, decay-limited |
| Oracle write timing, decay=0.999 | 99.0%, holds to gap 400 | -- (n/a) | decay is fully fixable given correct writes |
| Learned gate, decay=0.999 (no aux), **constant LR** | 6.3% | 0.99x | **superseded -- this was an LR artifact, not a selectivity ceiling (see below)** |
| Learned gate, warm-start / curriculum, constant LR | 1.3-1.7% | ~1.0-1.05x | unsupervised nudges don't work *under constant LR specifically* |
| Learned gate + aux BCE, decay=0.95 | 0.7-1.8% | **1.24-1.64x** | selectivity learnable even under fast decay, but doesn't help accuracy |
| Learned gate + aux BCE, decay=0.999 (single seed, 2000 ep, constant LR) | 26.4% | 1.77x | superseded (non-representative single seed) |
| Learned gate + aux BCE, decay=0.999 (5 seeds, 2000 ep, constant LR) | mean 19.0%, std 10.8%, range 1.8-33.1% | mean 1.58x, std 0.14x | selectivity is stable; accuracy under constant LR is not |
| Learned gate + aux BCE, decay=0.999, 6000-ep LR-decay schedule (2 seeds) | 43.9%, 76.7% | -- | superseded by the schedule-duration finding below -- not a general result |
| **No-aux**, decay=0.999, matched 9000-ep LR-decay schedule (2 seeds) | **45.0%, 52.6%** | **1.29-1.61x** | **the dominant effect in this entire phase is the LR schedule, not gate supervision** |
| **Aux BCE**, decay=0.999, matched 9000-ep LR-decay schedule (2 seeds) | **53.8%, 57.7%** | -- | aux supervision's real, isolated effect: **+5-9pp over no-aux**, not +40-75pp |
| All four 9000-ep LR-decay configs above | still rising at ep9000 (3 of 4 fail the 2pp convergence check) | -- | **none of these numbers are converged; true asymptotic values are unknown** |

## Interpretation

The Phase 2 conclusion ("AAC-diff's linear relaxation loses to attention,
cause unclear, maybe the update rule") is now six separated, evidenced
claims instead of one vague one -- two hold up unchanged, one holds up
only in a much weaker form than originally stated, and three needed real
revision after the checks above:

**Holds up unchanged:**

1. The read/retrieval side was never the problem -- the earlier
   "dual bottleneck" reading of the perfect-write test doesn't survive
   joint training and should be considered superseded.
2. The fixed `decay=0.95` schedule was silently capping the achievable
   memory horizon regardless of gate quality; this is a solved problem
   (`decay->0.999`, or a learned/adaptive decay, closes it). Confirmed
   unaffected by the engine fixes found later in this phase (Robustness
   check, item 1).

**Does not survive in its original form:**

3. ~~Gate selectivity is a second, real, dominant bottleneck under fast
   decay, and it is not a soft optimization difficulty that yields to
   easier initialization or curriculum scaffolding.~~ **This is wrong as
   stated.** The completely unsupervised, no-aux gate learns real
   selectivity (ratio 1.29x-1.61x, clearing every threshold that
   previously defined "selective" in this document) with zero
   architectural changes, purely by fixing the learning-rate schedule
   (see "Isolating the LR-decay effect from gate supervision"). It *was*
   a soft optimization difficulty, for a large part of its effect. What
   survives: auxiliary supervision still adds a further, smaller, real
   boost on top of a properly-optimized baseline (roughly +5-9pp
   accuracy at a matched schedule, seeIsolating section) -- a genuine but
   much more modest effect than "the dominant bottleneck."

**Needed real revision:**

4. Direct supervision on the gate *is* capable of teaching real,
   *low-variance* selectivity -- but the accuracy it produces is
   *high-variance*: across 5 seeds, held-out accuracy ranged 1.8%-33.1%
   (std 10.8pp) while gate ratio stayed tight (1.39x-1.74x, std 0.14).
   The original single-seed "26.4%" is not a reliable characterization
   of what this configuration does under constant LR. (Under LR decay,
   both aux and no-aux variance patterns still need their own multi-seed
   check -- not yet done; see caveats.)
5. Selectivity and decay do interact rather than contributing
   independently (the `decay=0.95` aux runs proved that: real
   selectivity, zero accuracy benefit) -- but selectivity and *accuracy*
   also do not move together even once decay is fixed, which originally
   looked like an open question about the architecture. **It has since
   been resolved, and resolved in favor of the more boring explanation:**
   it was optimization instability from a constant, too-high learning
   rate, not an architectural mystery, and it is *not specific to
   configurations with gate supervision* -- the plain no-aux gate shows
   the identical pattern (real selectivity, accuracy that depends
   entirely on whether the LR schedule lets the rest of the network
   converge). Seed 4's aux config went from 1.8% to 76.7% with only the
   LR schedule changed; seed 4's *no-aux* config goes from a similarly
   crippled constant-LR baseline (Finding 3: 6.3% at seed 0; seed 4 not
   separately measured under constant LR, but there is no reason to
   expect it behaves differently) to 45-65% depending on schedule length.
   This is no longer an open question about AAC-diff's architecture at
   all; it's a solved optimization-hygiene issue that happens to be far
   larger than the gate-supervision effect it was originally discovered
   alongside.
6. The apparent "meaningfully more headroom" found by sweeping
   hyperparameters at a fixed 2000-episode budget was largely a training-
   duration artifact under constant LR (the sweep's best cell peaked at
   28.4% then declined to 19.4% by episode 6000). Under LR decay, this
   picture is superseded again by the schedule-duration finding above:
   even "with LR decay" is not one fixed quantity -- a 6000-episode decay
   schedule and a 9000-episode decay schedule give substantially
   different numbers for the identical seed and config, and none of the
   numbers produced under either schedule have been confirmed converged.

**Bottom line -- revised a third time, and this time the revision cuts
against the phase's main causal claim, not just its precision:** the
dominant effect discovered in this entire investigation was never gate
selectivity -- it was that every experiment in Findings 1 through 5 (and
Phase 2 before them) was run with a constant, unscheduled Adam learning
rate, and that alone was capping achievable accuracy at chance-to-single-
digits regardless of gate quality, decay, or supervision. Fixing the
schedule alone, with zero gate supervision, recovers 45-65% held-out
accuracy depending on how long the schedule runs. Auxiliary gate
supervision remains a real, separable, positive contributor on top of
that fix (+5-9pp at a matched schedule, consistent across the two seeds
tested) -- but it is a secondary effect, not the primary one, and
Finding 3's framing of it as "the dominant bottleneck" should be
retracted, not merely revised upward. **Nothing in this document should
yet be cited as "AAC-diff reaches X% on MQAR."** No configuration --
aux or no-aux -- has been run to confirmed convergence; the LR schedule
itself has only been tested in two arbitrary shapes (6000- and
9000-episode linear decay) out of an unexplored space that likely
matters as much as anything else tested in this phase; and the Phase 2
attention baseline this document keeps comparing against has never once
been given the same optimization treatment being credited here. Until
all three of those are addressed, the honest summary of Phase 3 is:
*this architecture's differentiable relaxation was being evaluated under
a badly-tuned optimizer for its entire investigation, that turned out to
matter more than the architectural question the phase set out to answer,
and the architectural question itself (does gate supervision help, and
by how much) is now answerable in principle but has only been checked at
two seeds on two schedules.*

above as the single most important next experiment) has not been run.
Nothing currently in this document should be treated as this
investigation's final word on what makes AAC-diff work.

## Honest caveats / what this doesn't show

- **Superseded by the Robustness check above:** the episode-budget and
  non-convergence caveats from the original Finding 4 are now resolved
  (see items 2 and 4) -- and resolved unfavorably for the original point
  estimate. Keeping this line here so the history is legible: the
  original text read "the 25.3%/26.4% number should be read as a lower
  bound, not a ceiling." That framing turned out to be wrong in an
  interesting way -- it wasn't a lower bound on a monotonically-improving
  quantity, it was one sample from a wide and non-monotonic distribution.
- Aux-loss supervision requires the oracle write-position labels
  (`key_positions` from `task_mqar.py`), which are diagnostic-only
  ground truth not available in a real deployment setting -- this
  result shows the gate *can* learn selectivity given a training signal
  for it, not that AAC-diff solves MQAR unsupervised. A faithful next
  step is a *proxy* signal that doesn't require oracle labels (e.g. the
  original non-differentiable prototype's approach: `aac/controller.py`
  trains its utility MLP on a cheap proxy label -- "is this token a
  KEY/VAL event" -- rather than ground-truth write positions; an
  analogous proxy for MQAR's shared-vocabulary setting is an open
  question since there's no separate KEY/VAL token range to key off of).
- **New, and now the central open question:** why does gate selectivity
  (ratio) stay stable across seeds/configs while accuracy does not? At
  least three data points (multi-seed's seed 4, the sweep's
  `gate_w=0.5,pos_w=5` cell, and the general shape of the sweep grid)
  show respectable ratios (1.5-1.6x) paired with near-chance accuracy.
  Candidate explanations not yet tested: gate *magnitude* (not just the
  key/filler ratio) may matter -- the failing cases tend to have lower
  absolute gate values even when the ratio is fine; or the read pathway's
  own convergence might be seed-sensitive independent of the gate; or
  this could be a symptom of the same optimization instability flagged
  in Robustness check item 4 (no LR schedule) rather than a distinct
  phenomenon.
- **New:** none of this phase's runs use a learning-rate schedule --
  constant Adam `lr=1e-2` throughout, in every experiment across all
  three phases. The peak-then-decline dynamics in Robustness check item 4
  are consistent with this being a real gap, not yet tested.
- The hyperparameter sweep (Robustness check item 3) is single-seed;
  given the multi-seed variance found in item 2, the exact ranking of
  the top few grid cells should not be trusted to better than the noise
  floor established there (~11pp std). The qualitative pattern
  (`pos_weight=2` outperforming higher values) is repeated across
  multiple rows and is more likely to be real than the precise ordering
  of any two adjacent cells.
- All numbers throughout this phase remain single-seed *except* where
  explicitly noted as multi-seed above (Robustness check item 2 only).
  A full multi-seed x hyperparameter-grid x extended-training study is
  the natural, and now clearly necessary, next step before any number
  from this phase goes into a paper.

## Methods note: three autodiff engine issues found during this phase

None are specific to the gate-selectivity investigation -- all three are
general properties/defects of `aac/autodiff.py` that predate this phase
-- but all three directly affected the ability to *run* or *trust* the
experiments above, so they're recorded here for provenance:

1. **Reference-cycle memory leak.** Every op's `_backward` closure
   captures `out` (the tensor it's attached to) from its enclosing scope,
   creating a direct reference cycle on every non-leaf `Tensor` --
   potentially thousands per episode. Plain refcounting can't free a
   cycle; only the generational cyclic GC can, and cost grows as cyclic
   garbage accumulates, which showed up as training getting steadily
   slower over a run (up to ~30x slowdown observed on local hardware over
   ~2000 episodes) rather than any correctness error. Fixed by explicitly
   breaking the cycle (`_backward = None` on non-leaf nodes) once a graph
   is done being used -- inside `Tensor.backward()` for training calls,
   and via a new standalone `ad.free_graph()` for eval-mode calls (which
   never call `.backward()` and so never hit the training-side fix).
2. **Recursion-depth crash on long sequences.** The topological sort
   underlying both `backward()` and `free_graph()` was recursive
   (one Python stack frame per graph node). Fixing bug (1) meant
   eval-mode calls started actually traversing their graphs for the
   first time, which surfaced this: long episodes (gap=300-400 -> 350-450+
   tokens, several chained ops each) build dependency chains thousands of
   nodes deep, well past Python's default recursion limit (1000),
   producing a `RecursionError`. Fixed by converting the traversal to an
   iterative, explicit-stack version (`ad._topo_sort`), which has no
   depth ceiling other than available memory. Verified against episodes
   up to ~450 tokens with no crash, and against the full gradient-check
   suite (all ops + the 3-step BPTT/gated-memory case, ~1e-11 precision)
   to confirm the traversal change didn't alter any computed gradient.
3. **Non-deterministic child-iteration order (found resolving the gate-
   ratio/accuracy decoupling above; fixed in the LR-decay isolation work
   that followed).** `Tensor._prev` was a plain `set(_children)`;
   `Tensor` defines no custom `__hash__`/`__eq__`, so it fell back to
   Python's identity-based default, and a set's iteration order over
   identity-hashed objects depends on their memory addresses -- not fixed
   by any RNG seed, and not guaranteed stable across separate process
   runs. Since floating-point addition isn't exactly associative, this
   changed the order gradients from multiple parents accumulate into a
   shared child, and that bit-level noise compounded over thousands of
   training episodes into visibly different trained models, even with
   every explicit seed in the codebase held fixed (confirmed directly:
   rerunning the same 12-config sweep with identical seeds gave different
   numbers each time). Unlike (1) and (2), this never caused an incorrect
   result in any single computation -- it was a reproducibility gap, not
   a correctness bug -- but it meant no "same-seed rerun" anywhere in this
   project's history had been verified bit-identical before this was
   found. **Fixed** by giving every `Tensor` an incrementing creation-order
   id at construction and replacing the set with a list of unique children
   sorted by that id. Verified two ways: the full gradient-check suite
   still passes to ~1e-10 (confirming the fix changed traversal order
   only, not any computed value), and the same seed/config run twice in
   separate process invocations now produces bit-identical results to 15
   decimal places (previously confirmed non-identical under the same
   test). Anyone re-running experiments from a copy of the engine older
   than this fix should expect run-to-run drift from identical seeds.

All three fixes are in the `autodiff.py`/`models_diff.py` versions
underlying the "Isolating the LR-decay effect from gate supervision"
section above; the numbers in every section before it (everything from
Finding 1 through the original "Resolving the gate-ratio/accuracy
decoupling" section) were produced without fix (3) in place and should
be read with that caveat.


## Files added/changed this phase

```
aac/autodiff.py        added bce_loss op (gradient-checked to ~1e-11);
                        fixed a reference-cycle memory leak in
                        Tensor.backward() and added ad.free_graph() for
                        the eval-mode path; converted the recursive
                        topological sort to an iterative ad._topo_sort()
                        to remove a RecursionError on long episodes
                        (see "Methods note" above)
aac/models_diff.py      AACDiffModel: gate_mode="oracle_decay"/"curriculum",
                        gate_bias_init, aux_gate_weight, aux_pos_weight;
                        wired ad.free_graph() into eval-mode _finish paths
run_oracle_vs_learned.py    Finding 1 experiment
run_gate_fixes.py           Finding 3 experiment (warm-start, curriculum)
run_decay_sweep.py          Finding 2 + 3 experiment (decay x gate_mode grid)
run_aux_gate.py              Finding 4 + 5 experiment (aux BCE sweep)
run_multiseed.py             Robustness check item 2 (5-seed replication)
run_sweep.py                 Robustness check item 3 (hyperparameter grid)
                              (item 1's Phase 2 reproduction and item 4's
                              extended-training runs were one-off scratch
                              scripts, not saved as standalone files -- the
                              methodology is fully described in the
                              Robustness check section above and is a
                              straightforward rerun of train_compare.py /
                              run_aux_gate.py at longer episode counts)
```
