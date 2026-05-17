import torch
import os
import pandas as pd
from tqdm import tqdm


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a.flatten(), b.flatten()) / (a.norm() * b.norm())


model = "ViT-B-16"
checkpoints_dir = f"artifacts/checkpoints-analysis-drift/{model}"
results_dir = f"artifacts/results-analysis-drift/{model}"
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
for d in tqdm(datasets, desc="datasets"):
    task_dir = os.path.join(checkpoints_dir, d + "Val")
    covariance_filenames = sorted(
        [f for f in os.listdir(task_dir) if "covariance_" in f],
        key=lambda f: int(f.split("_")[-1].replace(".pt", "")),
    )

    c_final = None
    for filename in tqdm(covariance_filenames[::-1], desc=d, leave=False):
        c_t = torch.load(os.path.join(task_dir, filename))
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
