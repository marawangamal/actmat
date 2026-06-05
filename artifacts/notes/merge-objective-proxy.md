# Toward a merge objective that is a better proxy than the RegMean loss

_2026-06-05 — working note_

## TL;DR

The **RegMean loss is a poor proxy for merge accuracy.** Across t5/ViT models the
ACTMat minimizer attains a *higher* (worse) RegMean loss than the plain mean
("identity"), yet ACTMat merges to *higher* downstream accuracy. So minimizing the
RegMean objective does not select the better-merging method — it rewards the
conservative one.

**Goal:** find a different objective `J(Δ)` that (i) correlates with downstream
merge accuracy, and (ii) under which the **ACTMat estimator (`dᵀd`) attains lower
error than identity**. Bonus if `J` is data-free and has a closed-form minimizer
(so it also yields a merge method).

## The paradox (evidence)

Per-layer RegMean loss is `L(Δ; {cᵢ}) = Σᵢ ‖Δ − dᵢ‖²_{cᵢ}`, scored against the
**true** covariance `cᵢ`. Each method picks `Δ* = (Σᵢ dᵢ ĉᵢ)(Σᵢ ĉᵢ)⁺` from its
covariance surrogate `ĉ` (actmat: `dᵀd`; identity: `I`; regmean: true `c`).

**t5-base** — accuracy vs. summed RegMean loss flatly disagree:

| method | merge accuracy ↑ | Σ RegMean loss ↓ |
|---|--:|--:|
| actmat | **0.797** | 3.72e6 (worst) |
| regmean | 0.774 | **3.85e5** (floor) |
| identity (mean) | 0.632 | 2.71e6 |

ACTMat is **best** on accuracy but **worst** on the RegMean loss; identity is the
reverse. `regmean` is the analytic loss floor by construction but only middling on
accuracy.

**Summed RegMean loss, all models** (lower = better; from `rm_loss_general.csv`):
actmat > identity on **t5-base, t5-large, ViT-B-16/32, ViT-L-14** (i.e. identity
"wins" the objective), and only actmat < identity on **Olmo-3-7b**. So the
objective does not even consistently rank the two, let alone match accuracy.

## Why the RegMean loss fails as a proxy

The estimator is judged by how close its merge `Δ*({ĉ})` lands to the **true-cov
merge** `Δ*({c})` (= regmean) in the `c`-metric. Three problems:

1. **It is governed by the inverse `(Σĉ)⁺`, not by how good `ĉ` is as a covariance
   estimate.** ACTMat's `dᵀd` is a *better* covariance estimate than `I`
   (per-expert cosine ≈ 0.30–0.50 vs ≈ 0.06–0.15 across models), but `Σ dᵀd` is
   ill-conditioned, so `(Σ dᵀd)⁺` **overshoots** and the merge lands far from the
   optimum → high loss. (`cov-estimate-error` / the `cosine_similarity` metric show
   actmat is the better estimate; the `inv_*` metrics show its inverse is the problem.)

2. **The penalty is asymmetric — it punishes aggression, rewards conservatism.**
   With `B = (Σc)⁺`, the applied inverse `B̂` can be too small (undershoot → bounded,
   "do-nothing" loss) or too big (overshoot → unbounded loss). identity's `(1/T)I`
   undershoots (safe); actmat's `(Σ dᵀd)⁺` overshoots (`inv_rel_l2 ≈ 1000×`). The
   RegMean loss therefore systematically favors the timid merge.

3. **The target is wrong.** `regmean` (true `c`) is the loss minimizer *by
   definition*, but it is **not** the accuracy-optimal merge (it trails ACTMat on
   t5-base accuracy). "Distance to the regmean solution" is simply not "merge
   quality," so any estimator that is judged by it inherits the wrong objective.

A clean illustration (2×2 toy, `Di=2`, 2 experts): with `ĉᵢ = sᵢsᵢᵀ` aligned to
`cᵢ` (per-expert cosine 0.999), the merge still loses to the mean because `Σĉ` is
near-singular where `Σc` is not — high cosine, bad merge, invisible to the estimate
metric. (Reproduced in this session; see the conditioning discussion.)

## Desiderata for a new objective `J`

1. **Proxy quality:** monotone (or at least correctly ordered) w.r.t. downstream
   merge accuracy across methods — in particular `J(actmat) < J(identity)`,
   reversing the current anti-correlation.
2. **Data-free / cheap:** ideally expressible from `{dᵢ}` (and `dᵀd`) alone, no
   held-out data — that is ACTMat's premise.
3. **Robust to inverse-conditioning and scale:** must not reward mere conservatism;
   should value *using* covariance structure, which is what helps accuracy.
4. **Closed-form / differentiable minimizer** (optional but valuable): then `J`
   also *defines* a merge method, not just a scorer.

**Validation:** we already have downstream accuracy per method (t5/ViT/OLMo eval
pipelines). For each candidate `J`, compute its per-method value and correlate with
accuracy; the winning `J` ranks `actmat ≥ regmean > identity` (matching accuracy),
unlike the RegMean loss.

## Candidate directions

- **(A) Regularize the solver, not (only) the objective.** The failure is largely
  the *unregularized* pseudo-inverse. Use a ridge solve `(Σĉ + λI)⁻¹`: small `λ`
  tames actmat's overshoot. `actmat-identity-inv` (already implemented) is the
  `λ→∞` extreme (`(1/T)I`); sweep `λ` and re-measure the RegMean loss — does actmat
  beat identity for some `λ`? If so, "RegMean loss + ridge" may already be a better
  proxy, and the fix is the solver. **Cheapest first experiment.**

- **(B) Score the estimator, not the merge.** A merge = (estimator `ĉ`, solver).
  The `cosine(ĉ, c)` estimate-quality metric *already* ranks actmat > identity. If
  estimate quality is the right proxy for accuracy (test this correlation), then the
  objective is "covariance-estimation error," and the solver should be chosen
  separately to be well-conditioned (see A).

- **(C) Held-out / generalization error.** Split the samples used to form the cov;
  fit the merge on split 1, evaluate `Σ‖Δ − dᵢ‖²` on split-2 covariances. Penalizes
  overfitting the in-sample inverse; ACTMat's structured `dᵀd` may generalize better
  than a brittle data-cov inverse.

- **(D) Interference / output-space objective.** Minimize the change each expert
  sees in its *outputs* under the merge, or cross-task interference
  `Σ_{i≠j} f(Δ, task j)`. ACTMat-style whitening is designed to reduce interference,
  so an interference objective should favor it; this is closer to "what the merged
  model actually does."

- **(E) Magnitude-aware inverse diagnostic as objective.** `inv_norm_ratio =
  ‖(Σĉ)⁺‖/‖(Σc)⁺‖` cleanly separates undershoot (`<1`, safe) from overshoot (`≫1`,
  fatal). Penalizing `max(ratio, 1/ratio)` or similar could turn the conditioning
  insight into a usable regularizer/score.

- **(F) Direct task-loss proxy.** A few held-out examples → loss or gradient
  alignment of the merged model. Most faithful to accuracy, least data-free; useful
  as the *ground-truth* others are validated against.

## Open questions

- Is "good covariance estimate" (cosine) actually a better accuracy proxy than
  "RegMean loss"? Quick to check against the eval numbers — if yes, that reframes
  the whole thing as estimator-quality + well-conditioned solver.
- Why is OLMo the exception (actmat already wins the RegMean loss there)? Different
  conditioning of `Σ dᵀd` (3 experts, larger `Dₒ`) vs vision/language (7–8 experts)?
- Does ridge `λ` recover an objective whose minimizer is both accurate and
  loss-optimal — i.e. is the solver the whole story?

## Pointers

- `scripts/analysis/rm-loss-general.py` — per-(model, layer, method) metrics:
  `rm_loss`, `cosine_similarity`, `inv_cosine_similarity`, `l2_error`,
  `inv_l2_error`, `rel_l2_error`, `inv_rel_l2_error` → `artifacts/analysis/rm-loss/rm_loss_general.csv`.
- `scripts/analysis/cov-estimate-error.py` — cosine of estimated vs true cov
  (subsumed by the above).
- t5-base merged accuracies: `artifacts/results/t5-base/group-main/merged/{actmat,regmean,mean}/metrics.json`.
