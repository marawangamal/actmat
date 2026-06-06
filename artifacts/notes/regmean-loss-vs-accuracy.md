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
scores up to 2. Implemented as `angular_rm_loss` in `rm-loss-general.py` for all
four methods.

**Result (it flips T5 and collapses ViT's gap).** Summed loss, ACTMat vs identity:

| model | L2 sum: actmat<id? | angular sum: actmat<id? | angular median(act/id) |
|---|:--:|:--:|--:|
| Olmo-3-7b | ✅ | ✅ | 0.89 |
| t5-base | ❌ | ✅ **flips** | 0.78 |
| t5-large | ❌ | ✅ **flips** | 0.65 |
| ViT-B-16 | ❌ (1.90×) | ❌ (1.14×) | 1.14 |
| ViT-B-32 | ❌ (1.62×) | ❌ (1.11×) | 1.11 |
| ViT-L-14 | ❌ (1.63×) | ❌ (1.08×) | 1.08 |

Removing the magnitude penalty kills the `wo`/overshoot hijacking: the angular
**sum** now agrees with the per-layer verdict (ACTMat wins OLMo + both T5, matching
accuracy) instead of being dominated by a few overshooting layers. ViT goes from
*catastrophically* worse (1.6–1.9×) to *marginally* worse (1.08–1.14×). So most of
ViT's ACTMat "loss" on the RegMean loss was **overshoot, not mis-direction** — but a
small residual directional gap remains, and ACTMat still beats mean on ViT
*accuracy*, so the angular loss is a much better proxy than L2 yet still not perfect
on vision (residual → the [merge-objective-proxy](merge-objective-proxy.md)
interference / output-space candidates).

## Finding 3 — head-mean ablation: the loss-dominating layers are accuracy-irrelevant

Direct test: re-merge with the catastrophic layers forced to a plain **mean**
(everything else unchanged), under a separate `group-headmean` (t5,
`DenseReluDense.wo`) / `group-8-headmean` (ViT FFT, `mlp.c_fc`). If those layers
drove the L2-loss reversal *and* mattered for accuracy, mean-merging them should
move accuracy a lot.

**t5 (FFT), accuracy:**

| model | method | group-main | group-headmean | Δ |
|---|---|--:|--:|--:|
| t5-base | actmat | 0.7974 | 0.7927 | −0.47 pp |
| t5-base | regmean | 0.7744 | 0.7629 | −1.16 pp |
| t5-large | actmat | 0.8320 | 0.8271 | −0.49 pp |
| t5-large | regmean | — | 0.8210 | — |
| t5-base | mean (ref) | 0.6323 | — | — |
| t5-large | mean (ref) | 0.5164 | — | — |

The `wo` layers carry **50–66% of the entire L2 RegMean-loss sum** yet are worth only
**~0.5–1.2 pp of accuracy.** ACTMat-headmean (0.79) still far exceeds plain mean
(0.63): **ACTMat's accuracy advantage does not come from the layers where it loses
the RegMean loss** — it comes from the `wi`/attention layers where it *wins* the
loss. The L2 sum is dominated by accuracy-irrelevant layers; that is the proxy
failure, made concrete.

**ViT (FFT), `c_fc`→mean, accuracy:**

| model | method | group-8 | group-8-headmean | Δ |
|---|---|--:|--:|--:|
| ViT-B-32 | actmat | 0.8287 | 0.8366 | **+0.80 pp** |
| ViT-B-32 | regmean | 0.8302 | 0.8021 | −2.82 pp |
| ViT-L-14 | actmat | 0.9217 | 0.9141 | −0.76 pp |
| ViT-L-14 | regmean | 0.9006 | 0.8802 | −2.04 pp |
| ViT-B-32 | mean (ref) | 0.6544 | — | — |
| ViT-L-14 | mean (ref) | 0.7928 | — | — |

For ViT **actmat**, neutralizing `c_fc` (81–87% of the L2 loss sum) moves accuracy
only a little and inconsistently — *helps* B-32 (+0.80, so actmat's `c_fc` overshoot
was mildly harmful, partly agreeing with the loss) but slightly *hurts* L-14
(−0.76). So `c_fc` carries a bit more accuracy signal than t5's `wo`, but still
modest (<1 pp) relative to its loss dominance, and actmat-headmean stays far above
plain mean (0.84 vs 0.65; 0.91 vs 0.79). RegMean *relies* on its `c_fc` merge
(−2 to −2.8 pp when removed).

NB: the LoRA eval is a no-op here — LoRA adapts only attention, so `c_fc` has zero
delta under LoRA (verified ‖Δ c_fc‖=0 for `lora_finetuned`, 0.45 for `finetuned`);
the c_fc effect only exists in the full finetune, which is what the rm-loss analysis
used. Hence `group-8-headmean` is run in FFT mode.

## Pointers

- `scripts/analysis/rm-loss-general.py` — now emits `angular_rm_loss` alongside
  `rm_loss` and the cov-estimate metrics → `artifacts/analysis/rm-loss/rm_loss_general.csv`.
- Per-layer / per-type aggregation reproduced in this session (median & geomean of
  the per-layer actmat/identity ratio; layer-type Σ-share).
- Head-mean override: `--mean-keys <substr>` (src/args.py) forces matching layers to
  a plain mean merge in `combine_task_vectors` (src/merging.py), keeping `--merge-func`
  for the rest. Drivers: `scripts/{language,vision}/eval_task_addition_headmean.sh`.
- Merge accuracies: `artifacts/results/<model>/group-{main,headmean,8,8-headmean}/merged/{actmat,mean,regmean}/`.
