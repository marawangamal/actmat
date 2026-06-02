# Experiments

## Per-layer-type merge attribution (the "layer × method" grid)

### Goal

Find **which merge method is best for which layer type** in a model like
Olmo-3-7B. The standard pipeline applies one method uniformly to every 2D
weight, but attention vs. MLP layers may have different merge geometries.

### Grid

- **rows** `i` = transformer linear layer type
- **cols** `j` = merge method (`isoc`, `tsv`, `regmean`, `actmat_herm`, …; `mean` is the background, not a column)

Layer types under test (Olmo-3-7B): `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, `down_proj`. **`lm_head` and `embed_tokens` are always
`mean`-merged** (ignore-mean, never the method under test) — they sit in the
background bucket with the 1D norms, in every cell and baseline.

### Cell semantics

**Cell `(i, j)`:** merge layer type `i` with method `j`; merge *everything else*
(other transformer layers, norms, `lm_head`, `embed_tokens`) with `mean`. The
cell value is the merged model's **minerva_math_500** score.

- Down a column: which layer type benefits most from method `j`.
- Across a row: which method is best for layer type `i`.
- Above the all-mean floor ⇒ the method helps that layer type; below ⇒ it hurts.

### Baselines

- **All-mean** — every layer merged with `mean` (incl. head/embed). The floor every cell shares except its one deviating layer type.
- **All-method `j`** — every transformer layer merged with `j` (the v2 full merge). "Method everywhere" vs. "method on one layer type".

### Metric

minerva_math_500 scored by **`math_verify`** (symbolic equivalence), *not*
`metrics.json`'s `primary_score`: on `::tulu` runs the strict `exact_match`
collapses to ~0 (LaTeX/`\boxed{}` + the 512-tok cap). `math_verify` is
per-instance only in `task-001-minerva_math_500-predictions.jsonl`; the cell
value is its mean over the 500 predictions.

### Implementation — assemble cells on the fly, never recompute

The merge in `src/hf/merge.py` is **per-layer independent**: each layer's merged
weight depends only on that layer's tensors. So a cell's weights are *already
computed*:

- type-`i` layers → exactly their values in the **full method-`j` merge**
- everything else (incl. `lm_head`/`embed_tokens`) → their values in the **full `mean` merge**

Only **two sources** per cell, both already on disk.

⇒ **Do not re-run merges.** Reuse the per-method full merges from v2
(`.../group-rl-zero/merged/{method}`); build any missing **once**.

**Why a naive symlink view fails.** vLLM (and HF) use a checkpoint's
`index.json` only to decide *which* `.safetensors` files to open; once a file is
open they load **every** tensor in it ([weight_utils.py](../.venv-olmo/lib/python3.11/site-packages/vllm/model_executor/model_loader/weight_utils.py)
`for name in f.keys()`). So symlinking two *whole* shards that share keys (e.g.
`tsv` shard-1 and `mean` shard-1, which both contain every layer 0…k weight)
loads each shared key twice — last file wins — and the hybrid collapses to
all-one-method. The index is **not** consulted per-tensor.

**Fix — one file per layer (`group-rl-zero-hybrid/`).** Split each full merge into
single-key files so no two files ever share a key:

```
group-rl-zero-hybrid/
├── experts/{method}/<key>.safetensors      # COPIES: one tensor per file (split_expert.py)
│     e.g. mean/model.layers.0.self_attn.q_proj.weight.safetensors, …  (all 355 keys, every method)
└── merged/{layer}_{method}-ct-math/         # one dir per cell — SYMLINKS only (assemble_cell.py)
      ├── <key>.safetensors -> ../../experts/{method|mean}/<key>.safetensors   # 355 links
      ├── model.safetensors.index.json       # generated: key -> its own filename
      └── config.json + tokenizer + chat_template.jinja   # copied from a ref merge
```

A cell symlinks the `--layer-type` keys from `--method` and all other keys from
`mean`. Each opened file yields exactly one key, so the union over all 355 files
is the full model with **no duplicates** — a clean hybrid, no per-source
renaming needed (each key is taken from only one source). `experts/` is the only
real-bytes cost (one full-model copy per method); cells are pure symlinks, so the
whole grid stays on disk — **no teardown**. The one lifetime constraint:
`experts/` must not be moved/deleted while any cell points into it.

Layer type of a key = its second-to-last dotted component
(`…q_proj.weight` → `q_proj`, `lm_head.weight` → `lm_head`).

### Scope

Methods (columns): `isoc`, `tsv`, `actmat_herm`, with `mean` as the
background/floor. Layer types (rows): the 7 transformer projections. Each cell is
one `minerva_math_500::tulu` eval of its assembled hybrid dir.

Scripts: `scripts/hybrid/split_expert.py` (full merge → `experts/{method}/`),
`scripts/hybrid/assemble_cell.py` (`(layer, method)` → a cell dir of symlinks +
index + config).
