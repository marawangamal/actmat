# Is the RegMean loss really lower for averaging than for ACTMat?

_2026-06-05 — working note. Companion to [merge-objective-proxy.md](merge-objective-proxy.md)._

## Central question

**Why does ACTMat reach higher test accuracy than averaging, even though using the
identity covariance in the RegMean minimizer (= averaging) gives a lower RegMean
loss?** Two sub-questions drive this note:

1. Is the observation even *correct* — does identity really attain a lower RegMean
   loss than ACTMat? (The summed table below says yes for everything except OLMo.)
2. If so, *why*, and is there a different (e.g. angular) objective under which ACTMat
   wins — one that would be a better proxy for merge accuracy?

## The summed table (what started the paradox)

Total RegMean loss `Σ_layers Σ_i ‖Δ − d_i‖²_{c_i}` (lower = better):

| method | Olmo | ViT-B-16 | ViT-B-32 | ViT-L-14 | t5-base | t5-large |
|---|--:|--:|--:|--:|--:|--:|
| regmean | 3,591 | 55,792 | 16,443 | 128,924 | 384,960 | 5,866,220 |
| identity | 31,899 | 224,633 | 75,028 | 553,552 | 2,705,357 | 16,276,489 |
| actmat | 27,587 | 518,143 | 88,565 | 1,064,209 | 3,724,085 | 22,927,424 |
| actmat-identity-inv | 57,266 | 286,128 | 95,468 | 697,572 | 3,174,582 | 33,384,076 |

By the **sum**, ACTMat beats identity only on OLMo. This is what looks paradoxical.

## Finding 1 — the sum is a scale artifact; per-layer, ACTMat usually wins

The sum adds L2 RegMean losses across layers spanning 768²→11008² and wildly
different delta scales, so a *handful* of large, ill-conditioned layers dominate it.
On a **per-layer** basis (the unit at which "is this merge better" is meaningful):

| model | median(actmat/identity) | geomean | layers ACTMat wins | sum-implied ratio |
|---|--:|--:|--:|--:|
| Olmo-3-7b | **0.69** | 0.79 | **72%** | 0.87 |
| t5-base | **0.81** | 0.87 | **60%** | 1.38 |
| t5-large | **0.66** | 0.69 | **74%** | 1.41 |
| ViT-B-16 | 1.90 | 2.11 | 19% | 2.31 |
| ViT-B-32 | 1.62 | 1.60 | 24% | 1.18 |
| ViT-L-14 | 1.63 | 1.66 | 26% | 1.92 |

So for **OLMo and both T5 models, ACTMat already attains a lower RegMean loss than
averaging on the majority of layers** (median ratio < 1, 60–74% win rate). The
*summed* table reverses T5 only because the L2 sum is hijacked by a few layers.
**ViT is the genuine exception** — there ACTMat loses even per-layer.

## Finding 2 — the catastrophic layers are the wide-input FF / down-projections

Identity-sum share and ACTMat win-rate, by layer type:

| model | dominant layer type | share of Σ-loss | ACTMat win% there | median ratio there |
|---|---|--:|--:|--:|
| t5-base | `DenseReluDense.wo` (FF-out, 3072→768) | 50% | **4%** | 2.36 |
| t5-large | `DenseReluDense.wo` (FF-out, 4096→1024) | 66% | 17% | 1.69 |
| Olmo | `mlp.down_proj` (FF-out, 11008→4096) | 42% | 59% | 0.83 |
| ViT-* | `mlp.c_fc` (FF-**in**, 768→3072) | 81–87% | 17–37% | 1.3–2.1 |

The sum is concentrated in the **widest-input** linear layers — the FF
down-projections (`wo`, `down_proj`) for T5/OLMo. Large input dim `D_i` ⇒ a
`D_i×D_i` covariance that is the most ill-conditioned ⇒ `(Σ d⊤d)⁺` **overshoots**
hardest ⇒ a catastrophic L2 RegMean loss on exactly those few layers. On T5 the
rest of the network (`wi`, attention `o`, `q`) is an ACTMat landslide (win
80–100%), but `wo` alone outweighs all of it in the sum.

OLMo escapes the reversal because its `down_proj` overshoot is milder (still 59%
win, median ratio 0.83), so the sum stays in ACTMat's favor.

For ViT the loss is *not* a one-layer artifact: ACTMat loses on `c_fc` **and** loses
100% of attention `q/k/v`, so it is genuinely worse on the RegMean L2 loss there.

## Interim answer to the central question

The premise "identity gets a lower RegMean loss than ACTMat" is **mostly an artifact
of summing a scale-sensitive L2 loss across heterogeneous layers.** Measured per
layer, the RegMean loss already *agrees* with accuracy on OLMo and T5 (ACTMat wins),
and the reversal is driven by a few wide-input FF-out layers where the unregularized
pseudo-inverse overshoots. The L2 RegMean loss punishes that overshoot
quadratically and without bound, even when the merge direction is correct.

**ViT remains the real disagreement**: ACTMat is worse on the RegMean L2 loss
(even per-layer) yet better on accuracy. So a better proxy must reverse ViT too.

## The angular RegMean loss (test in flight)

If the catastrophic layers are *overshoot* — right direction in the `c_i`-metric,
wrong magnitude — then a **scale-invariant angular** objective should not punish
them:

```
L_ang(Δ) = Σ_i ( 1 − cos_{c_i}(Δ, d_i) ),   cos_{c_i}(x,y) = <x,y>_{c_i} / (‖x‖_{c_i} ‖y‖_{c_i})
```

`L_ang` ignores `‖Δ‖` entirely, so a 2× overshoot scores 0, while a wrong direction
scores up to 2. Implemented as `angular_rm_loss` in `rm-loss-general.py` and added
as a metric for all four methods (job 9743557).

**Prediction:** `L_ang(actmat) < L_ang(identity)` on OLMo/T5 by a wider margin, and
— the real test — possibly on ViT too, *if* ViT's ACTMat loss is overshoot rather
than mis-direction. If ViT still loses under `L_ang`, then on ViT ACTMat's RegMean
*direction* is genuinely worse and accuracy is tracking something the RegMean
geometry (any norm) does not capture — pointing back to the
[merge-objective-proxy](merge-objective-proxy.md) candidates (interference /
output-space / held-out objectives).

## Pointers

- `scripts/analysis/rm-loss-general.py` — now emits `angular_rm_loss` alongside
  `rm_loss` and the cov-estimate metrics → `artifacts/analysis/rm-loss/rm_loss_general.csv`.
- Per-layer / per-type aggregation reproduced in this session (median & geomean of
  the per-layer actmat/identity ratio; layer-type Σ-share).
- Merge accuracies: `artifacts/results/<model>/group-*/merged/{actmat,mean,regmean}/`.
