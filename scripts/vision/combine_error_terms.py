import argparse
import os

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="ViT-B-16")
args = parser.parse_args()

base = f"artifacts/results-analysis/{args.model}"
corr = pd.read_csv(os.path.join(base, "error_corr_term.csv"))
cross = pd.read_csv(os.path.join(base, "error_cross_term.csv"))

corr["type"] = "corr"
cross["type"] = "cross"
true_rows = pd.concat([corr, cross], ignore_index=True)
true_rows["mode"] = "true"

ctrl_rows = pd.read_csv(os.path.join(base, "error_terms.csv"))
ctrl_rows = ctrl_rows[ctrl_rows["mode"] == "ctrl"]

out = pd.concat([true_rows, ctrl_rows], ignore_index=True)
out_path = os.path.join(base, "error_terms_submission.csv")
out.to_csv(out_path, index=False)
print(f"Wrote {len(out)} rows to {out_path} (true={len(true_rows)}, ctrl={len(ctrl_rows)})")
