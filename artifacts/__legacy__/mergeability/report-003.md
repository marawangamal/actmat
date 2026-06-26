# Report 003 — acting on the predictor: control fails, selection works (Phase 3)

**Question.** Report-002 showed the data-free residual $r_t(\Delta_m(S))$ predicts
per-merge retention (within-task $\rho_s \approx -0.8$). Can we *act* on it — (a) by
reweighting the merge to protect threatened tasks, (b) by selecting which experts to
co-merge?

## (a) Residual-reweighted ACTMat — null result

**Method.** Scale each task's covariance in the actmat solve:
$W^* = (\sum_t w_t \Delta_t \hat C_t)(\sum_t w_t \hat C_t)^+$ with
$w_t = T\cdot\mathrm{softmax}(r_t/\tau)$ from the unweighted merge's residuals
(`merge_rw.py`). τ tuned **data-free** on predicted residuals: α=1.0 dominated
unweighted (predicted worst-task residual −39%, mean −9%). Evaluated on the same 36
subsets as phase 1 (paired; 36 l40s jobs).

**Result: predicted gain did not materialize.**

| paired per subset (n=36) | actmat | actmat_w |
|---|---:|---:|
| worst-task retention (avg) | 0.824 | 0.828 |
| mean retention (avg) | 0.919 | 0.919 |
| subsets where actmat_w better (worst-task) | — | 18/36 |

**Interpretation (Goodhart gap).** $r_t$ correlates with retention across subsets
because it measures *which tasks interfere*; but inside a fixed subset the merge must
still represent all members in one delta — reallocating measured residual between
tasks via the $\hat C$ metric is ≈ zero-sum in function space. The residual is an
excellent *observable*, not a control knob: prediction ≠ control.

## (b) Subset selection — works

Using only phase-1 data (no new runs): rank same-size subsets by predicted worst
residual $\max_t r_t(S)$ and compare to realized worst-task retention under actmat.

| subset size | $\rho_s(\max_t r_t,\ \min_t \mathrm{ret})$, n=12 |
|---|---:|
| M=4 | **−0.73** |
| M=8 | **−0.84** |
| M=12 | −0.38 |

Concrete: among the twelve M=8 subsets, the best-predicted has worst-task retention
0.898; the worst-predicted 0.811 — an 8.7-point worst-case gap identified from weights
alone, before building any merged model.

## Takeaway / next

The covariance residual should be used **upstream of the merge** (compatibility
screening, partitioning a pool of experts into co-merge groups, deciding what *not*
to merge), not inside it. Redirected question: **data-free merge planning** — given
$N$ experts and a budget of $k$ merged models, partition by predicted residuals to
maximize worst-task retention; validate the chosen vs random/adversarial partitions
with a handful of evals. (M=12's weaker ρ suggests the signal compresses as subsets
approach the full pool — planning matters most at small-to-medium group sizes.)

**Artifacts.** `phase1_weights_ViT-B-32.csv` (α=1.0 softmax weights),
`phase1_weighted_predictions_ViT-B-32.csv` (τ grid), `subsets/ViT-B-32/actmat_w/`
(36 evals). Code: `merge_rw.py`, `predict_weighted_residuals.py`,
`eval_subset_merge.py` (actmat_w branch).
