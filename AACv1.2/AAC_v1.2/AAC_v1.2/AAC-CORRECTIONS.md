# AAC — Math Corrections

This document lists corrected versions of three issues found in the original AAC specification (`AAC.txt`): a circular dependency in the unified state-update loop, a sign error in the memory credit-assignment rule, and two notation collisions. Each fix preserves the original intent and is a minimal change — everything else in the source document (attention read, hierarchical routing complexity, merge rule, overall objective) is unaffected and composes correctly with these corrections.

---

## 1. Circular dependency in the update loop (original §22–23)

**Problem.** The original loop computed the query and the controller action using `S_t`, but `S_t` itself was only defined afterward as a function of `a_t`:

```
q_t = Q(e_t, S_t, T_t)          # uses S_t
a_t = π(S_t, T_t, P_t)          # uses S_t
S_t = F_S(S_{t-1}, x_t, a_t)    # S_t defined here, using a_t
```

This is self-referential: `a_t` depends on `S_t`, which depends on `a_t`. As written, the system has no valid evaluation order.

**Fix.** Make the query and controller act on the *previous* fast state, `S_{t-1}`, and only advance to `S_t` once `a_t` is known. Every right-hand side now references only quantities with a strictly earlier or already-resolved timestamp:

```
e_t   = E(x_t, S_{t-1})                              # event formation, uses prior state
q_t   = Q(e_t, S_{t-1}, T_{t-1})                      # query built from prior state
a_t   = π(S_{t-1}, T_{t-1}, P_{t-1})                  # policy acts on prior state
m_t   = Read(P_{t-1}, q_t)                            # retrieval uses prior memory
S_t   = F_S(S_{t-1}, x_t, a_t)                        # advance fast state
T_t   = λ_t · T_{t-1} + F_T(x_t, S_t)                 # trace may use the new S_t
h_t   = F_H(x_t, S_t, T_t, m_t)                       # output — all dependencies resolved
ΔP_t  = Validate(T_t, a_t)
P_t   = Merge(Evict(P_{t-1} + ΔP_t))
```

`T_t` is allowed to use the freshly computed `S_t` because nothing else in this timestep depends back on `T_t` — that's the one safe same-step dependency. `q_t` and `a_t` are not.

*Alternative if tighter reactivity to the current input is required:* split the fast-state update into a partial pre-update used for routing, `Ŝ_t = F_S^{partial}(S_{t-1}, x_t)`, followed by the full update `S_t = F_S(Ŝ_t, a_t)` after the action is chosen. This adds a sub-step not present in the original document, so the one-step-lag version above is the minimal correction.

---

## 2. Sign error in memory credit assignment (original §21)

**Problem.**

```
C_i = Σ_t α_{t,i} (L_base − L_ablated(i))
U_i ← U_i + η C_i
```

If memory `i` is genuinely useful, removing it should *increase* loss (`L_ablated(i) > L_base`), which makes `(L_base − L_ablated(i))` **negative**. Under `U_i ← U_i + ηC_i`, a useful memory would then have its utility estimate *decreased* — the opposite of the intended behavior, and inconsistent with the utility convention already established in §7 (`U = E[L_without − L_with]`, positive when retention helps).

**Fix.** Flip the subtraction order so `C_i` is positive exactly when ablating the memory hurts performance:

```
C_i = Σ_t α_{t,i} (L_ablated(i) − L_base)
U_i ← U_i + η · C_i
```

Now `C_i > 0` whenever removing memory `i` makes the model worse, so useful memories are reinforced and unhelpful ones decay toward eviction (`c_i < τ ⇒ evict(i)`), matching the "self-evaluating memory" behavior the section describes.

---

## 3. Notation collisions

Two symbols were reused for unrelated quantities within equations meant to compose together. The global state variables `S_t` (fast state) and `P_t` (persistent memory) are load-bearing everywhere in the document, so the *local, narrow-scope* variables are renamed instead.

**a) Low-rank transition factor (original §5)**

```
A_t = D_t + P_t Q_t^⊤,   rank(P_t Q_t^⊤) = r ≪ d
```

`P_t` here collides with persistent memory `P_t` from §4/§10. Renamed:

```
A_t = D_t + L_t Q_t^⊤,   rank(L_t Q_t^⊤) = r ≪ d
```

**b) Surprise term in the write score (original §8)**

```
w_t = α·Û_t + β·N_t + γ·S_t − δ·R_t
```

`S_t` here collides with the fast recurrent state. Renamed (surprise → `Ψ_t`):

```
w_t = α·Û_t + β·N_t + γ·Ψ_t − δ·R_t
```

---

## Summary

| Issue | Location | Fix |
|---|---|---|
| Circular `S_t` / `a_t` dependency | §22–23 unified loop | Query and policy act on `S_{t-1}`, not `S_t` |
| Sign error in credit assignment | §21 | `C_i = Σ α_{t,i}(L_ablated − L_base)` |
| `P_t` collision (memory vs. low-rank factor) | §5 | Rename factor to `L_t` |
| `S_t` collision (fast state vs. surprise) | §8 | Rename surprise to `Ψ_t` |

With these four changes, the full state-space system composes without circularity, self-inconsistent updates, or symbol collisions. All other equations in the source document — the attention-style read (`α_i = softmax(q_t^⊤k_i)`, `m_t = Σα_iv_i`), the hierarchical routing complexity (`O(d(log M + K))`), the confidence-weighted merge rule, and the overall `E[ΔL] > cost` decision principle — are unaffected and can be substituted in as-is.
