import torch
import os
import pandas as pd


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a.flatten(), b.flatten()) / (a.norm() * b.norm())


model = "ViT-B-16"
checkpoints_dir = f"checkpoints-analysis-drift/{model}"
results_dir = f"results-analysis-drift/{model}"
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
    covariance_filenames = sorted(
        [
            f
            for f in os.listdir(os.path.join(checkpoints_dir, d + "Val"))
            if "covariance_" in f
        ],
        key=lambda f: int(f.split("_")[-1].replace(".pt", "")),
    )

    c_final = None
    # DEBUG:
    print(f"Covariance order: {covariance_filenames[::-1]}")
    for filename in covariance_filenames[::-1]:
        c_t = torch.load(os.path.join(checkpoints_dir, d + "Val", filename))
        c_final = c_t if c_final is None else c_final
        step = int(filename.split("_")[-1].replace(".pt", ""))
        for layer in c_final:
            if layer.endswith("_n"):
                continue
            rows.append(
                {
                    "dataset": d,
                    "layer_name": layer,
                    "step": step,
                    "cosine_similarity": cosine_similarity(c_t[layer], c_final[layer]).item(),
                    "type": "cross",
                }
            )

df = pd.DataFrame(rows)
df.to_csv(os.path.join(results_dir, "error_terms.csv"), index=False)
print(f"Saved results to {os.path.join(results_dir, 'error_terms.csv')}")
