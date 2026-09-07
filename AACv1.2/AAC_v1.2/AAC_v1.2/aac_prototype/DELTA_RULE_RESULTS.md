> **v1.2 note:** The gate-selectivity conclusions in this document (real
> selectivity failure, no update rule fixes it) are the starting point
> for the full investigation in `PHASE3_GATE_RESULTS.md`, which finds a
> much larger confound (an unscheduled learning rate) sitting underneath
> everything reported here. Read this file for historical context on how
> the delta-rule variants were ruled out; don't treat its numbers as the
> final word on gate selectivity.

# Delta-Rule Implementation Results

## Summary
Implemented two delta-rule variants for AAC-diff's memory mechanism in addition to the original Hebbian update. Neither variant improved performance on the MQAR benchmark, confirming that **the bottleneck is not the memory update rule, but the model's failure to learn selective writing**.

## Implementations Tested

### 1. Regularized Delta-Rule (Original Approach)
**Memory update:**
$$M_t = \text{decay} \cdot M_{t-1} + g_t \cdot \text{outer}(k, v) - g_t \cdot \beta \cdot \text{outer}(k, k^T M_{t-1})$$

**Result:** No improvement. Still ~2.2% accuracy.

### 2. Widrow-Hoff Delta-Rule (Second Approach)
**Memory update:**
$$M_t = M_{t-1} + g_t \cdot \text{outer}(k, v - k^T M_{t-1})$$

**Result:** No improvement. Still ~0.8% accuracy (slightly worse).

## Benchmark Results (MQAR Task)

| Model | Held-out Accuracy | Gap 20-40 | Gap 300-400 |
|-------|-------------------|-----------|------------|
| RNN-only | 1.0% | 1.7% | 1.0% |
| **Attention** | **57.2%** | **57.1%** | **17.7%** |
| Hebbian (original) | 2.2% | 2.3% | 1.5% |
| Regularized delta-rule | 2.2% | 2.3% | 1.5% |
| Widrow-Hoff delta-rule | 0.8% | 0.6% | 2.9% |

## Key Finding: Gate Selectivity Failure

The **write gate never learns to be selective**:

| Update Rule | Key Positions | Filler Positions | Ratio |
|-------------|--------------|-----------------|-------|
| Hebbian | 0.371 | 0.437 | 0.85x |
| Regularized delta-rule | 0.371 | 0.437 | 0.85x |
| Widrow-Hoff delta-rule | 0.412 | 0.493 | 0.84x |

**Target:** Gate should be higher at KEY positions (ratio > 1.0), but all variants have ratio < 0.9x.

## Why Delta-Rule Didn't Fix the Problem

1. **The model never learns what to write in the first place** - the gate doesn't distinguish keys from fillers
2. **Attention learns via softmax's built-in selectivity**
3. **AAC-diff's gate has no such structure** - it's just `sigmoid(W_g @ h)`

## Conclusion

Phase 2 correctly identified that the memory update rule wasn't the primary bottleneck. The continuous relaxation of AAC's memory loses the discrete routing logic that made selective writing possible.

The experiment successfully validates Phase 2's conclusion: **a literal continuous relaxation of AAC's memory is not competitive with softmax attention on this task**.

## Files Modified

1. **aac/autodiff.py** - Added `sub` operation with gradient checking
2. **aac/models_diff.py** - Added delta-rule memory update implementations
3. **test_autodiff.py** - Added tests for subtraction and delta-rule BPTT (all pass ~1e-11 precision)

---

## Corrected Diagnostic (Off-by-One Bug Fix)

### The Bug Discovery
During Phase 4 re-analysis, an **off-by-one indexing bug** was identified in the original gate selectivity diagnostic:
- **Original error:** Compared gate values at KEY token positions against filler positions
- **Actual timing:** The memory write happens ONE position LATER (when the value token appears), so the gate should be evaluated at position `key_position + 1`, not `key_position`

### Corrected Gate Diagnostic Results (with fixed indexing)

| Update Rule | Gates at KEY→VALUE writes | Gates at FILLER positions | Ratio |
|-------------|--------------------------|--------------------------|-------|
| Hebbian | 0.208 | 0.227 | **0.92x** |
| Regularized delta-rule | 0.469 | 0.463 | **1.01x** |
| Widrow-Hoff delta-rule | 0.464 | 0.483 | **0.96x** |

Even with corrected indexing, none of the three update rules achieve selective writing. This confirms the original conclusion: **the bottleneck is gate learning, not the update rule.**

### Perfect-Write Test Results (Read Mechanism Isolation)

| Update Rule | Perfect-Write Accuracy | Chance Baseline | Status |
|-------------|------------------------|-----------------|--------|
| Hebbian | 0.7% | 1.6% | Below chance |
| Regularized delta-rule | 3.3% | 1.6% | Barely above chance |
| Widrow-Hoff delta-rule | 1.3% | 1.6% | Below chance |

**Note (v1.2):** this "dual bottleneck" reading (read mechanism might also
be broken) does not survive joint training from scratch with an oracle
gate -- see `PHASE3_GATE_RESULTS.md` Finding 1. The read pathway here was
undertrained because the gate never fired during its own training, not
because linear dot-product retrieval is structurally incapable.

### Revised Conclusion (superseded further by Phase 3)

1. **Gate selectivity remains unlearned** across all three update rules
2. **Read mechanism fundamentally broken** — *(superseded, see note above)*
3. **Delta-rule variants unhelpful** because they assume learned associations that never materialize

The failure is **architectural, not algorithmic** — *(further revised by
Phase 3: substantially an optimization-hygiene failure, not purely
architectural; see `PHASE3_GATE_RESULTS.md`)*.
