# Report 001 — can ACTMat covariances predict mergeability? (Phase 0)

**Question.** When many experts are merged into one model, which tasks keep their
accuracy? Rahamim et al. 2026 (arXiv:2601.06672) found the dominant predictor is the
base model's zero-shot accuracy on the task (requires evals), while weight norms predict
nothing. Can ACTMat's data-free covariance $\hat C_t = \Delta_t^\top \Delta_t$ predict it from
weights alone?

**Experiment.** ViT-{B-32, B-16, L-14}, 20 full-FT experts each, all 20 previously
merged by each of 10 methods — everything from artifacts on disk, nothing new was run.

- **Target** (per task $t$, method $m$): $\;\mathrm{ret}_{t,m} = \mathrm{acc}_t(\mathrm{merged}_m) / \mathrm{acc}_t(\mathrm{expert}_t)$.
- **Scores** (per task $t$; $\Delta_t$ = expert task vector, $\Delta_m$ = actmat merged task vector):

| score | definition | needs |
|---|---|---|
| zs | base zero-shot accuracy on $t$ (paper's predictor) | evals |
| $r_t$ | $\dfrac{\sum_\ell \operatorname{tr}[(\Delta_t-\Delta_m)\hat C_t(\Delta_t-\Delta_m)^\top]}{\sum_\ell \operatorname{tr}[\Delta_t \hat C_t \Delta_t^\top]}$ — predicted disturbance of task $t$: 0 = merge imitates expert, 1 = no better than base | weights |
| $a_t$ | $\operatorname{mean}_{s\neq t} \cos_F(\hat C_t, \hat C_s)$ — covariance overlap with other tasks | weights |
| $\|\Delta_t\|_F$ | weight norm (paper's negative control) | weights |

- **Statistic**: Spearman $\rho_s(\text{score}, \mathrm{ret}_{\cdot,m})$ over the 20 tasks, per method
  ($|\rho_s| \ge 0.45 \Rightarrow p<0.05$).

**Results.** Per model: mean/std of $\mathrm{ret}_{t,m}$ over the 20 tasks, and
$\rho_s(\text{score}, \mathrm{ret}_{\cdot,m})$ per score. Methods ordered strong → weak.

### ViT-B-32

| method | mean ret | std ret | zs | $r_t$ | $a_t$ | norm |
|:---|---:|---:|---:|---:|---:|---:|
| actmat | 0.84 | 0.11 | 0.29 | −0.09 | −0.06 | −0.10 |
| ace | 0.86 | 0.11 | 0.31 | −0.08 | −0.08 | −0.08 |
| tsv | 0.84 | 0.11 | 0.28 | −0.09 | −0.06 | −0.26 |
| regmean | 0.83 | 0.14 | 0.62 | 0.29 | 0.14 | −0.32 |
| ties | 0.65 | 0.18 | 0.62 | 0.20 | 0.28 | −0.20 |
| mean | 0.69 | 0.23 | **0.91** | **0.65** | **0.66** | −0.16 |

### ViT-B-16

| method | mean ret | std ret | zs | $r_t$ | $a_t$ | norm |
|:---|---:|---:|---:|---:|---:|---:|
| actmat | 0.87 | 0.13 | 0.29 | −0.02 | −0.01 | −0.19 |
| ace | 0.88 | 0.13 | 0.33 | −0.02 | 0.00 | −0.14 |
| tsv | 0.86 | 0.12 | 0.38 | 0.04 | 0.05 | −0.35 |
| regmean | 0.85 | 0.15 | 0.61 | 0.33 | 0.23 | −0.36 |
| ties | 0.71 | 0.18 | 0.58 | 0.22 | 0.31 | −0.20 |
| mean | 0.71 | 0.22 | **0.85** | **0.61** | **0.54** | −0.27 |

### ViT-L-14

| method | mean ret | std ret | zs | $r_t$ | $a_t$ | norm |
|:---|---:|---:|---:|---:|---:|---:|
| actmat | 0.93 | 0.06 | 0.41 | −0.03 | 0.24 | 0.13 |
| ace | 0.95 | 0.05 | 0.38 | −0.06 | 0.18 | 0.03 |
| tsv | 0.93 | 0.06 | 0.39 | −0.11 | 0.20 | −0.08 |
| regmean | 0.91 | 0.09 | 0.52 | 0.02 | 0.09 | −0.26 |
| ties | 0.79 | 0.15 | 0.56 | 0.07 | 0.45 | 0.14 |
| mean | 0.77 | 0.22 | **0.93** | **0.52** | **0.48** | −0.11 |

Heatmaps: [ViT-B-32](corr_heatmap_ViT-B-32.png) · [ViT-B-16](corr_heatmap_ViT-B-16.png) ·
[ViT-L-14](corr_heatmap_ViT-L-14.png) — scatters and per-task CSVs alongside in this
directory.

The pattern is the same in every model:

1. **zs replicates, for weak merges only**: $\rho_s$ = 0.85–0.93 (mean), ≈ 0.6
   (ties/regmean), ≤ 0.41 n.s. (actmat/ace/tsv).
2. **Strong merges flatten mergeability**: retention std ≈ 0.22 (mean) → 0.06–0.13
   (actmat), shrinking with model size; for strong methods *no* score predicts the
   remainder (whole rows ≈ 0).
3. **$r_t$ and $a_t$ mirror zs without evals** (0.5–0.65 where zs is ≈ 0.9; ≈ 0 where zs
   is ≈ 0; $\rho_s(r_t, \mathrm{zs})$ = 0.79/0.59 on B-32/L-14) — they recover the paper's
   predictor from weights alone, but don't beat it. The norm control fails everywhere,
   as in the paper.

**Next** (PLAN.md, Phase 1): the paper's actual mergeability score — each expert merged
with random subsets over many trials (~640 single-task evals). There $r_t$ varies with
the subset and can make per-merge predictions static zs cannot.
