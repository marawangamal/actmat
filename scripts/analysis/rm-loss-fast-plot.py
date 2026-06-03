"""Per-layer RegMean-loss plots for the vision/language models (ViT-B-16, t5-base).

Reads artifacts/analysis/rm-loss/rm_loss_vl.json (written by rm-loss-fast.py) and,
per model, saves a wide log-scale line plot (rm_loss_<model>.png) and a total-loss
bar plot (rm_loss_<model>_total.png). Mirror of rm-loss-olmo-plot.py.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
N_SKIP = 2  # keep every N_SKIP-th layer (per model)
METHODS = ["actmat", "regmean", "identity"]

# Per-model layer-name abbreviations (applied in order). Unknown models fall
# back to just stripping ".weight".
REPLACE_BY_MODEL = {
    "t5-base": {
        "transformer.encoder.block.": "E.",
        "transformer.decoder.block.": "D.",
        "SelfAttention.": "SA.",
        "EncDecAttention.": "CA.",
        "DenseReluDense.": "FF.",
        ".weight": "",
    },
    "ViT-B-16": {
        "model.visual.transformer.resblocks.": "B.",
        "visual.transformer.resblocks.": "B.",
        ".attn.": ".attn.",
        ".mlp.": ".mlp.",
        ".weight": "",
    },
}


def line_plot(sub, out, title, n_skip):
    """Wide log-scale per-layer line plot for one model, one line per method."""
    g = sns.relplot(
        sub, x="layer_idx", y="metric", kind="line",
        hue="method", style="method", markers=True, aspect=4,
    )
    g.set(yscale="log")
    g.set_axis_labels("layer", "RegMean loss (vs. true covariance)")
    g.figure.suptitle(title, y=1.03)

    ticks = sub.drop_duplicates("layer_idx").sort_values("layer_idx")
    ticks = ticks[ticks["layer_idx"] % n_skip == 0]
    g.ax.set_xticks(ticks["layer_idx"])
    g.ax.set_xticklabels(ticks["tick_label"], rotation=45, ha="right", fontsize=7)
    # faint vertical guides dropping from each tick
    g.ax.grid(axis="x", color="gray", linewidth=0.4, alpha=0.25)
    g.ax.set_axisbelow(True)
    g.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def total_bar(sub, out, title):
    totals = sub.groupby("method")["metric"].sum().reindex(METHODS).dropna()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(totals.index, totals.values, color=sns.color_palette()[: len(totals)])
    ax.set_yscale("log")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.bar_label(bars, fmt="%.3g", padding=3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    base = REPO_ROOT / "artifacts/analysis/rm-loss"
    df = pd.DataFrame(json.loads((base / "rm_loss_vl.json").read_text()))

    for model in df["model"].unique():
        sub = df[df["model"] == model].copy()

        sub["layer_abbrv"] = sub["layer"]
        for old, new in REPLACE_BY_MODEL.get(model, {".weight": ""}).items():
            sub["layer_abbrv"] = sub["layer_abbrv"].str.replace(old, new, regex=False)
        sub["tick_label"] = (
            sub["layer_abbrv"] + "\n(" + sub["Do"].astype(str)
            + ", " + sub["Di"].astype(str) + ")"
        )
        # contiguous order in state-dict appearance order (matches OLMo: blocks in order)
        sub["layer_idx"] = pd.factorize(sub["layer"])[0]

        total_bar(
            sub, base / f"rm_loss_{model}_total.png",
            f"Total RegMean loss across layers ({model})",
        )
        line_plot(
            sub[sub["layer_idx"] % N_SKIP == 0],
            base / f"rm_loss_{model}.png",
            f"Layer-wise RegMean loss ({model})",
            n_skip=1,  # already subsampled above; label every kept point
        )


if __name__ == "__main__":
    main()
