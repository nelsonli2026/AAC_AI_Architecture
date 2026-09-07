# AAC — Adaptive Associative Computing

**AAC** is a proposed neural sequence architecture built around one idea: not all information deserves the same computational treatment. Instead of forcing a single recurrent hidden state to remember everything, AAC splits memory by **timescale, persistence, and value**, and lets a learned controller decide — at every step — what's worth keeping, retrieving, or computing further.

> **AAC is a controlled neural dynamical system that continuously decides what to keep, what to retrieve, what to forget, and how much computation to spend, according to the predicted future value of information.**

**v1.2 note:** see `aac_prototype/PHASE3_GATE_RESULTS.md` for where the
empirical investigation of this idea currently stands. Short version: not
yet validated as competitive with attention, and the investigation is
still open (see that document's final section for the recommended next
experiment).

---

## Core principle

Every decision in AAC — read, write, promote, merge, evict, reason — is governed by a single economic rule:

```
Expected future value of information  >  cost of storing or processing it
```

This turns the architecture into a resource-allocation problem rather than a fixed stack of layers.

## State: three timescales

AAC represents system state as a triple:

```
Z_t = (S_t, T_t, P_t)
```

| Component | Role | Timescale | Main cost |
|---|---|---:|---:|
| **S_t** — fast state | Continuous, compressed recent context (SSM / linear-attention style) | Very short | Continuous compute |
| **T_t** — temporary trace | Buffer where uncertain information accumulates evidence before being trusted | Medium | Small recurrent cost |
| **P_t** — persistent memory | Long-term associative store for information judged durably valuable | Long | Storage / retrieval |

Information flows through increasing persistence: `token → S_t → T_t → P_t`, with promotion happening only when there's enough evidence of long-term value.

## How it works, end to end

1. **Encode** the input into a structured event (entity/relation/value/time/confidence, plus a residual to avoid losing information to a narrow bottleneck).
2. **Estimate utility** — a learned predictor scores how much the event would reduce future loss if retained.
3. **Update the fast state** continuously (subquadratic, ~`O(N·d²)`, avoiding full quadratic attention).
4. **Decide whether to retrieve** from persistent memory — via a hierarchical/clustered index (`O(d(log M + K))` instead of a brute-force `O(M·d)` scan), with the number of parallel queries scaling with uncertainty.
5. **Update the temporary trace** with decay, so information that never accumulates evidence naturally fades.
6. **Promote to persistent memory** only when justified — writes are sparse (a fraction `ρ ≪ 1` of inputs, not all of them).
7. **Maintain memory as evidence, not fact** — items carry confidence and provenance, so contradictory information can coexist as competing hypotheses (e.g. `P(B|A)=0.6` vs `P(C|A)=0.4`) rather than being silently overwritten.
8. **Reinforce, merge, and evict** — memories gain confidence when reused, compatible memories merge (confidence-weighted average), contradictory ones stay separate, and low-confidence memories are removed.
9. **Reason adaptively** — the number of extra reasoning steps scales with input difficulty/uncertainty rather than being fixed.

All of this is driven by one policy:

```
a_t = argmax_a [ E(ΔL | a) − λ_C·C(a) − λ_M·M(a) ]
```

i.e., take an action only when its expected reduction in loss outweighs its compute/memory/retrieval cost.

## Why: the problem with full attention

Full attention gives broad access to a sequence but at growing interaction cost. AAC instead routes information by its likely lifetime:

- Recent context → recurrent state
- Potentially durable information → persistent memory
- Ambiguous cases → extra (adaptive) computation

This means long-range retrieval doesn't require reconstructing the whole history at every step.

## Complexity target

```
O(N·d²)                          fast recurrent pathway
+ O(ρ·N·d·(log M + K))           sparse, hierarchically-routed memory ops
+ O(d · Σ_t n_t)                 adaptive reasoning
```

under the operating assumptions `ρ ≪ 1`, `K ≪ M`, `n_t ≪ N` for most inputs — i.e., the expensive mechanisms are meant to be *selective*, not universal, keeping `|P_t| ≪ N`.

## Relationship to existing ideas

- **KDA / GDN / state-space models** — provide the foundation for AAC's fast recurrent state (`S_t`); AAC's contribution is *not* forcing that state to also carry durable information.
- **Classical associative memory** — AAC externalizes a learned associative store (`P_t`), but adds learned policies for *when* to write, promote, query, merge, or forget it.
- **Adaptive computation** — the controller acts as a compute gate, spending more reasoning steps on harder inputs, making AAC closer to a resource-aware dynamical system than a fixed-depth network.

## Full state-space summary

```
e_t   = E(x_t, S_{t-1})                          # event formation
q_t   = Q(e_t, S_t, T_t)                          # query construction
a_t   = π(S_t, T_t, P_t)                          # controller policy
m_t   = Read(P_t, q_t)                            # associative retrieval
S_t   = F_S(S_{t-1}, x_t, a_t)                    # fast state update
T_t   = λ_t·T_{t-1} + F_T(x_t, S_t)               # temporary trace update
h_t   = F_H(x_t, S_t, T_t, m_t)                   # output
ΔP_t  = Validate(T_t, a_t)                        # candidate persistent write
P_t   = Merge(Evict(P_{t-1} + ΔP_t))              # persistent memory update
```

subject to the overall objective:

```
min_θ  E[ L_task + λ_C·C(a_t) + λ_M·M(P_t) + λ_R·R(q_t, P_t) + λ_W·W(a_t) ]
```

Note: `aac_prototype/AAC-CORRECTIONS.md` corrects a circular dependency
in this loop (query/policy must act on `S_{t-1}`, not `S_t`) -- the
prototype code implements the corrected version.

## One-line definition

```
AAC = multi-timescale recurrent state
    + learned information valuation
    + sparse associative memory
    + hierarchical retrieval
    + adaptive computation
    + memory competition
```

Goal: long-context capability + persistent memory + adaptive reasoning, without every token paying the full cost of global attention, dense memory retrieval, or deep computation.

---

*This README is a condensed summary of `AAC.txt`, a detailed mathematical specification of the AAC (AAC-1.1) architecture. It is a conceptual/theoretical design, not a benchmarked implementation — see `aac_prototype/` for the implementation and `aac_prototype/PHASE3_GATE_RESULTS.md` for where empirical validation currently stands.*
