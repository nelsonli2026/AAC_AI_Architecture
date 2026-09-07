> **v1.2 note:** Several conclusions in this document are revised or
> overturned by `PHASE3_GATE_RESULTS.md` -- most importantly the
> perfect-write test's "read mechanism might be independently broken"
> reading (see Phase 3 Finding 1), and this file's numbers were produced
> before three autodiff engine bugs were found and fixed (see Phase 3's
> Methods note). Read this file for historical context, not as current
> fact.

# Phase 2: baseline comparison, end-to-end differentiability, and a harder task

This extends the original prototype to actually test the three things it
couldn't test before: **(a)** a real baseline comparison at matched
parameter/compute budget, **(b)** a genuine attempt at end-to-end
differentiable training through the memory mechanism, and **(c)** a
harder task where the answer isn't hand-fed via pre-aligned embeddings.
Read this after the main `README.md` — it assumes that context.

**Bottom line up front:** the differentiable relaxation of AAC's memory
loses badly to plain softmax attention on a fair, matched-budget,
harder benchmark, and at full difficulty it doesn't learn the task at
all. That's the honest result. It's also not a surprising one in light of
published work on linear/fast-weight attention variants — see
"Interpreting the result" below.

## What changed to make this run

### (b) End-to-end differentiability

No PyTorch/JAX is available in this sandbox, so `aac/autodiff.py` is a
small hand-written reverse-mode autodiff engine (a "micrograd for
tensors"): a `Tensor` class with `.backward()`, and manually-derived
backward passes for each op (`matvec`, `vecmat`, `outer`, `dot`, `tanh`,
`sigmoid`, `softmax`, `softmax_cross_entropy`, `embedding_lookup`,
`concat`, `stack`, elementwise add/scale). Every op is verified against
numerical (finite-difference) gradients in `test_autodiff.py`, including
a 3-step recurrence with a gated fast-weight memory update — the exact
computational pattern the real models use — matching to ~1e-10 precision.

AAC's discrete write/promote/evict decisions can't be backpropped through
directly. Rather than bolt on REINFORCE or Gumbel-softmax (which bring
their own optimization headaches and would make it hard to tell what's
failing), the memory in this phase is a **continuous, gated associative
(fast-weight) matrix**:

```
M_t = decay * M_{t-1} + g_t * outer(k_t, v_t)
```

with `g_t = sigmoid(...)` a learned, differentiable write gate — the
relaxed version of AAC's discrete write decision. This is not a novel
idea; it's the same family as Fast Weight Programmers (Schlag et al.
2021) and the linear-attention lineage AAC's own document cites as
related work (KDA, GDN, DeltaNet). **This is a real scope decision, not
a hidden one**: it tests "AAC's organizing principle, made
differentiable," not the literal discrete architecture from the spec.

### (c) A harder, non-rigged task

`task_mqar.py` replaces the earlier synthetic task with the standard
**induction-head / multi-query associative-recall (MQAR)** format used in
real SSM-vs-attention papers (Based, H3, Zoology): one shared vocabulary,
no separate KEY/VALUE/QUERY token ranges, no privileged identity
embedding. A "query" is just an earlier key token literally reappearing.
Nothing hands the model a pre-built way to recognize "this token matches
that one" — it has to learn that from scratch, through the embedding
table and the key/value/query projections.

### (a) Matched-budget baselines

`aac/models_diff.py` defines three models, all sharing `d_emb=32`,
`d_h=32`, trained on identical episodes with identical Adam settings
(`lr=1e-2`, gradient accumulated over mini-batches of 8 episodes):

- **RNNBaseline** — plain tanh-RNN + linear readout, no external memory.
- **AttentionBaseline** — one-layer causal self-attention over the full
  token history. Stated plainly: this model's memory is **uncompressed**
  — it never forgets or throws anything away, so its capacity grows with
  sequence length. It's a ceiling, not a fair capacity match.
- **AACDiffModel** — fast recurrent state `h_t` (AAC's `S_t`) + the gated
  fixed-size associative memory `M_t` described above (AAC's `P_t`,
  relaxed). Unlike the attention baseline, `M_t` is a **fixed `d_h x
  d_h` matrix regardless of sequence length** — it has to actually
  compress, which is the entire point of AAC's design.

## A real bug found and fixed along the way

The first full run produced **chance-level accuracy for all three
models, including attention** — which shouldn't happen; softmax
attention reliably solves induction-head tasks. That's a "something is
broken" signal, not a result, so it was chased down rather than reported.

The bug: both the attention baseline and AAC-diff computed a token's key
*and* value from the *same* current token's representation. That means
attending back to an earlier key's position retrieved information about
the key itself, never the value that had followed it — the value was
never actually bound to the key's memory slot. Fixed by shifting the
binding by one position (`key = f(token[t-1])`, `value = f(token[t])`),
which is the standard induction-head circuit and is what makes "attend
back to an earlier occurrence of this token" recover the *next* token
rather than a copy of itself. After the fix, a quick small-scale test
went from ~14% to ~72% (attention) and ~97% (AAC-diff) accuracy. Also
switched from stepping the optimizer every single episode to
accumulating gradients over mini-batches of 8 (single-episode online SGD
was too noisy to learn efficiently).

This is reported here rather than silently fixed, because it's exactly
the kind of mistake that would have produced a false "the whole approach
doesn't work" conclusion if it had gone unnoticed.

## Results

Trained for 3,000 episodes each (`n_vocab=64`, 6 key/value pairs per
episode), matched dims and optimizer:

| Model | Held-out accuracy | Accuracy at gap 300–400 (unseen length) |
|---|---:|---:|
| RNN-only (no memory) | 1.4% | 1.5% |
| **Attention (uncompressed)** | **57.2%** | **17.7%** |
| AAC-diff (fixed-size gated memory) | 1.4% | 3.1% |

(Chance = 1.6%, since there are 64 possible values.)

- **RNN-only stays at chance**, as expected — a plain recurrent state
  has no mechanism for content-addressable lookup.
- **Attention learns the task solidly** (57% vs. 1.6% chance) and
  degrades gracefully, not catastrophically, at gap lengths 5-10x longer
  than anything seen in training (57% → 18%).
- **AAC-diff never escapes chance at this difficulty**, despite having
  the same training budget and a real, working gradient signal (verified
  via the autodiff gradient checks).

### Isolating why

Smaller-scale probing found AAC-diff's linear associative memory is
**brittle**, not just weak: it solves easy configurations well (2 pairs
from a 16- or 64-word vocabulary: 82-85% accuracy) but degrades
**inconsistently** as difficulty rises. Diagnostic instrumentation ruled
out the obvious failure modes: memory norms, read magnitudes, gate
values, and hidden state norms were all numerically well-behaved even in
configurations where accuracy was at chance. A post-hoc check for
emergent selectivity — does the learned write gate fire more strongly at
informative (key) positions than at filler positions — found **no
signal** at the full-scale setting (ratio 0.81x, i.e. no difference).

## Interpreting the result

This is not "the code doesn't work" — the autodiff is gradient-checked
to 1e-10, and the same mechanism (gated fast-weight memory) solves easy
versions of the task well. It's a genuine finding about the *specific*
differentiable relaxation used: an **unnormalized, no-softmax, plain
Hebbian-style associative write** doesn't provide a strong enough,
reliable enough training signal to separate many competing key/value
pairs, especially compared to softmax attention's naturally sharp
(exponentially-amplified) retrieval.

This actually lines up with published results: papers studying exactly
this benchmark (e.g. the Based / Zoology line of work) report that
**plain linear attention reliably underperforms softmax attention on
MQAR**, and that closing the gap requires additional structure — a
**delta-rule** update (subtract out the old association before writing a
new one, as in DeltaNet) rather than pure additive Hebbian accumulation.
AAC's own document cites exactly these enhanced variants (KDA, GDN,
DeltaNet) as its inspiration — not plain linear attention.

## Files added in this phase

```
aac/autodiff.py         hand-rolled reverse-mode autodiff (Tensor + ops)
test_autodiff.py        numerical gradient checks for every op, incl. full BPTT
task_mqar.py             harder MQAR/induction-head task, shared learned vocab
aac/models_diff.py       RNNBaseline, AttentionBaseline, AACDiffModel (matched budget)
train_compare.py         trains + evaluates all three, generalization sweep, gate diagnostic
```

## Honest answer to "is the architecture good"

Still not proven good, and by Phase 3, several parts of this
investigation's own methodology were found to need correction too (see
`PHASE3_GATE_RESULTS.md`) -- most notably, this phase's numbers were all
produced under a constant, unscheduled learning rate that turned out to
be a dominant confound. The organizing principle — route information by
predicted durability rather than treating everything uniformly — still
hasn't been tested in its most defensible form, with correct optimizer
hygiene, at scale.
