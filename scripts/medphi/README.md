# MediPhi — clinical merging experiment (CLUE eval)

Merge the 5 [MediPhi](https://huggingface.co/microsoft/MediPhi) clinical experts onto
`Phi-3.5-mini-instruct` and evaluate on the [CLUE](https://github.com/TIO-IKIM/CLUE)
benchmark ([Dada et al. 2024](https://arxiv.org/abs/2404.04067)). This reproduces the
**MediPhi** SLM results ([arXiv:2505.10717](https://arxiv.org/abs/2505.10717)).

MediPhi (`microsoft/MediPhi-Instruct`) is a 5-expert clinical merge of
`Phi-3.5-mini-instruct` (SLERP each expert back to base, then BreadCrumbs-TIES) — a
clean, tuned task-arithmetic baseline to compare ACTMat against.

## Scope / caveats

- **2 open CLUE tasks only**: `MeQSum` (3-shot, ROUGE/BERTScore) and `LongHealth`
  (0-shot, MC accuracy). The other 4 CLUE tasks (`MedNLI`, `Problem List Summary`,
  `MeDiSumQA`, `MeDiSumCode`) need **credentialed PhysioNet / MIMIC-IV** access.
- **CLUE+** (the 12-task superset the MediPhi paper reports its headline averages
  on) adds 6 datasets that are **not publicly released** in the CLUE repo.
- Decoding is **greedy** to match the paper (the upstream harness defaulted to
  `temperature=1.0` sampling).

## `clue.patch` — why each change to the CLUE harness

| Change | Reason |
|---|---|
| `.python-version` 3.12 → 3.11 | pandas 2.0.3 / nmslib have no cp312 wheels → source-build failures |
| `+datasets>=3.0` | old `datasets` (2.14) uses `pa.PyExtensionType`, removed in `pyarrow>=14` |
| `+transformers==4.45.2` | `>=4.46` blocks `torch.load` of `.bin` unless `torch>=2.6`; BERTScore's scibert is `.bin`-only and vllm 0.6.6 pins torch 2.5.1 |
| `max_model_len=16384` | Phi-3.5 advertises 131072 ctx → vLLM KV cache won't fit on an l40s |
| `temperature=0` | paper uses greedy decoding |
| `stop=["<|end|>","<|user|>"]` | some merged models' eos config omits `<|end|>` → run-on text tanks ROUGE |

## Run

```sh
bash scripts/medphi/setup.sh                          # clone CLUE + patch + build .venv-med + data
sbatch        scripts/medphi/eval_medphi_base.sh      # base Phi-3.5-mini-instruct
sbatch --array=0-4 scripts/medphi/eval_medphi_experts.sh   # the 5 MediPhi experts
sbatch --array=0-3 scripts/medphi/eval_medphi.sh      # merges (sum mean actmat tsv)
```

Each script sources the CLUE `.venv-med`, (merges via `src/hf/merge.py` for
`eval_medphi.sh`,) then runs the 2 open CLUE tasks. Results land in the standard
artifacts layout:

```
artifacts/checkpoints/Phi-3.5-mini-instruct/group-mediphi/merged/<method>/
artifacts/results/Phi-3.5-mini-instruct/group-mediphi/{pretrained,experts/<e>,merged/<method>}/{meqsum,longhealth8k}/results.json
```

(The array scripts stagger 90 s/task to avoid a jsonschema cold-start race across
the shared venv; LongHealth is slow, ~2-3 h/model.)

## Reproduction — MeQSum (Rouge-1 F1)

| Model | Ours (greedy) | Paper (Table 11) |
|---|---|---|
| `Phi-3.5-mini-instruct` | 36.0 | 36.7 |
| `MediPhi` | 38.0 | 37.9 |
| `MediPhi-Instruct` | 43.0 | 42.8 |

Matches within <1 point. LongHealth paper targets (LH, Table 11): base 45.9,
MediPhi 45.7, MediPhi-Instruct 45.0.
