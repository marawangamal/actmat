import torch
import os
import pandas as pd


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a.flatten(), b.flatten()) / (a.norm() * b.norm())


model = "ViT-B-16"
checkpoints_dir = f"checkpoints-analysis/{model}/max_batches_10"
results_dir = f"results-analysis/{model}"
os.makedirs(results_dir, exist_ok=True)

datasets = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]

rows = []
for d in datasets:
    for filename in os.listdir(os.path.join(checkpoints_dir, d + "Val")):
        if not "grad_cross" in filename:
            continue
        layer_name = filename.replace(".pt", "").replace(
            "grad_cross_matrix_model_visual_transformer_resblocks_", ""
        )
        gcm = torch.load(os.path.join(checkpoints_dir, d + "Val", filename))
        cosim_cross = cosine_similarity(gcm["gbar"].T @ gcm["gbar"], gcm["sbar"])
        cosim_corr = cosine_similarity(gcm["sbar"], gcm["stilde"])
        # TODO: add drift term
        # cov_terminal = torch.randn_like(gcm["sbar"])
        # cosim_drift = cosine_similarity(gcm["sbar"], cov_terminal)
        rows.extend(
            [
                {
                    "dataset": d,
                    "layer_name": layer_name,
                    "cosine_similarity": cosim_cross.item(),
                    "type": "cross",
                },
                {
                    "dataset": d,
                    "layer_name": layer_name,
                    "cosine_similarity": cosim_corr.item(),
                    "type": "corr",
                },
                # {
                #     "dataset": d,
                #     "layer_name": layer_name,
                #     "cosine_similarity": cosim_drift.item(),
                #     "type": "drift",
                # },
            ]
        )

df = pd.DataFrame(rows)
df.to_csv(os.path.join(results_dir, "error_terms.csv"), index=False)
print(f"Saved results to {os.path.join(results_dir, 'error_terms.csv')}")
