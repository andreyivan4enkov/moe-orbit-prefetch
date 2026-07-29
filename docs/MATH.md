# Mathematics of Object A (full formulas)

This document is the **math companion** to the source in `src/moe_orbit_prefetch/`.
Every formula below is implemented in code (file + function named). Edit the code freely; keep this doc in sync if you change equations.

Notation:

- \(h \in \mathbb{R}^d\) — residual / embedding vector (router-facing signal)
- \(E\) — number of routed experts (`n_experts`)
- \(k\) — top-k width (`top_k`)
- \(S \in \mathbb{R}^E\) — stigmergy field `S_env`
- \(W\) — history window length (`window`)

---

## 1. Normalize \(h\)

Code: `OrbitPredictor._unit`

\[
\hat h = \frac{h}{\|h\|_2 + \varepsilon}, \quad \varepsilon = \texttt{float64 eps}
\]

---

## 2. Local field scale

Code: `local_mean_threshold` / `OrbitPredictor._field_scale`

\[
\phi = \operatorname{mean}\{ S_i : S_i > 0 \}
\quad\text{(if empty: }\phi=\varepsilon\text{)}
\]

---

## 3. Predict scores (prefetch candidates)

Code: `OrbitPredictor.predict`

Start from the field:

\[
\texttt{scores} \leftarrow S
\]

### 3.1 Token induction

If `tok_id` was seen before with successor experts \(L(t)\):

\[
\forall e \in L(t):\quad \texttt{scores}[e] \leftarrow \texttt{scores}[e] + \phi
\]

### 3.2 History resonance

For each past step \(j\) in the last \(W\) deposits, with unit vector \(\hat h_j\) and expert set \(E_j\):

\[
c_j = \langle \hat h_j, \hat h \rangle
\]

Match floor (no magic constant):

\[
\tau = \max\bigl(0,\ \operatorname{mean}_j c_j\bigr)
\quad\text{(}\texttt{local\_match\_floor}\text{)}
\]

If \(c_j > \tau\):

\[
\forall e \in E_j:\quad \texttt{scores}[e] \leftarrow \texttt{scores}[e] + c_j
\]

### 3.3 Arg-top-k / cold start

If \(\max(\texttt{scores}) \le 0\) (cold):

\[
b = \big\lfloor |h_0|\cdot 10^6 \big\rfloor \bmod E, \quad
O = \{ (b+i)\bmod E : i=0..k-1 \}
\]

Else:

\[
O = \operatorname{arg\,top\text{-}k}(\texttt{scores})
\]

\(O\) is the **prefetch orbit** (not a claim that \(O\) equals the live gate).

---

## 4. Deposit (after true experts \(T\) from live gate)

Code: `OrbitPredictor.deposit`

Resonance from the window **before** writing the new event:

\[
R = \tau = \max\bigl(0,\ \operatorname{mean}_j c_j\bigr)
\]

Deposit scale (`deposit_scale`):

\[
\alpha = (1-R)\,(1 + \operatorname{mean}|S|) + \varepsilon
\]

Update masses:

\[
\forall e \in T:\quad S_e \leftarrow S_e + \frac{\alpha}{|T|}
\]

Decay / hold (`decay_hold_from_resonance`), only if history non-empty:

\[
S \leftarrow S \cdot \operatorname{clip}(R,\ \varepsilon,\ 1)
\]

Also store induction memory \(L(\texttt{tok\_id})\leftarrow T\) and append \((\hat h, T)\) to the window.
If history length exceeds stand size \(\texttt{window}\), keep exactly the last \(\texttt{window}\) pairs (no magic slack).

---

## 5. Prefetch horizon widening

Code: `OrbitPredictor.prefetch_set`

Width:

\[
k' = \min(E,\ k \cdot \texttt{copies})
\]

Same scoring as predict, with cosine contributions scaled by \(1/\texttt{copies}\); return top-\(k'\).

---

## 6. Emergent comparison (bench gate)

Code: `emerges_greater`

Given paired series \(a_i, b_i\):

\[
\begin{aligned}
\texttt{wins} &= \#\{i: a_i > b_i\}, \\
\texttt{losses} &= \#\{i: a_i < b_i\}, \\
\texttt{PASS} &\iff \texttt{wins} > \texttt{losses}
\end{aligned}
\]

Gap/MAD are diagnostics only — **not** PASS thresholds.

---

## 7. Expert residency sleep

Code: `DynamicExpertStore.evict_below_mean`

Let \(m_e\) be stigmergy mass for hot expert key \(e\). Threshold:

\[
\theta = \operatorname{mean}\{ m_e : m_e > 0 \}
\]

Evict (sleep) every hot expert with \(m_e < \theta\).

Hard trim: `trim_to_bytes(max_bytes)` repeatedly drops the lowest-\(m\) hot expert until resident bytes \(\le\) budget.

---

## 8. Runtime cycle (Object A)

Code: `SparseDeepseekRuntime` + store + predictor

On token \(t\) with hidden \(h_t\):

1. **Predict / prefetch** \(O_t = \mathrm{OrbitPredictor}(h_t)\) (async load slices).
2. **Exec** live MoE gate → true set \(T_t\); compute only experts in hot ∪ sync-miss.
3. **Deposit** \(S\) with \(T_t\).
4. **Sleep** cold experts via local mean / byte budget.

Live gate weights remain DeepSeek’s; our math governs **residency**, not the trained router logits.

---

## 9. What is intentionally *not* magic

Forbidden as quality PASS thresholds: hand-tuned margins like “hit > 0.15”.  
Allowed: machine \(\varepsilon\); structural model facts (e.g. 64 experts, top-6 on V2-Lite); stand size / seed.
