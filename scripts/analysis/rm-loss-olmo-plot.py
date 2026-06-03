"""Per-layer RegMean-loss plot for Olmo-3-7b (mirror of the t5/ViT analysis cell).

Reads artifacts/analysis/rm-loss/rm_loss_olmo.json (written by rm-loss-olmo.py) and saves
a wide log-scale line plot (one line per method) to artifacts/analysis/rm-loss/rm_loss_olmo.png.
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
N_SKIP = 4  # keep every N_SKIP-th tick label (225 layers is dense)

SUBMOD_ORDER = {  # within-block ordering
    "self_attn.q_proj": 0,
    "self_attn.k_proj": 1,
    "self_attn.v_proj": 2,
    "self_attn.o_proj": 3,
    "mlp.gate_proj": 4,
    "mlp.up_proj": 5,
    "mlp.down_proj": 6,
}
ABBRV = {
    "model.layers.": "L",
    ".self_attn.": ".",
    ".mlp.": ".",
    "q_proj": "Q",
    "k_proj": "K",
    "v_proj": "V",
    "o_proj": "O",
    "gate_proj": "G",
    "up_proj": "U",
    "down_proj": "D",
}


def sort_key(layer: str):
    m = re.match(r"model\.layers\.(\d+)\.(.*)", layer)
    if m:
        return (int(m.group(1)), SUBMOD_ORDER.get(m.group(2), 99))
    return (10**6, 0)  # lm_head et al. last


def abbrev(layer: str) -> str:
    s = layer
    for old, new in ABBRV.items():
        s = s.replace(old, new)
    return s


def block_of(layer: str) -> int:
    m = re.match(r"model\.layers\.(\d+)\.", layer)
    return int(m.group(1)) if m else 10**6  # lm_head et al. last


def line_plot(df, out, title, tick_every):
    """Wide log-scale per-layer line plot, one line per method.

    df must already carry a contiguous `layer_idx` and a `tick_label`; ticks are
    drawn on every `tick_every`-th distinct layer_idx.
    """
    g = sns.relplot(
        df,
        x="layer_idx",
        y="metric",
        kind="line",
        hue="method",
        style="method",
        markers=True,
        aspect=4,
    )
    g.set(yscale="log")
    g.set_axis_labels("layer", "RegMean loss (vs. true covariance)")
    g.figure.suptitle(title, y=1.03)

    ticks = df.drop_duplicates("layer_idx").sort_values("layer_idx")
    ticks = ticks[ticks["layer_idx"] % tick_every == 0]
    g.ax.set_xticks(ticks["layer_idx"])
    g.ax.set_xticklabels(ticks["tick_label"], rotation=45, ha="right", fontsize=7)
    # faint vertical guides dropping from each tick
    g.ax.grid(axis="x", color="gray", linewidth=0.4, alpha=0.25)
    g.ax.set_axisbelow(True)
    g.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def with_layout(df):
    """Attach contiguous `layer_idx` + `tick_label` for the layers present in df."""
    df = df.copy()
    order = sorted(df["layer"].unique(), key=sort_key)
    rank = {l: i for i, l in enumerate(order)}
    df["layer_idx"] = df["layer"].map(rank)
    df["tick_label"] = df.apply(
        lambda r: f"{abbrev(r['layer'])}\n({r['Do']}, {r['Di']})", axis=1
    )
    return df


def main():
    rows = json.loads(
        (REPO_ROOT / "artifacts/analysis/rm-loss/rm_loss_olmo.json").read_text()
    )
    df = pd.DataFrame(rows)
    base = REPO_ROOT / "artifacts/analysis/rm-loss"

    # full plot: all 225 layers, tick label every N_SKIP-th point
    line_plot(
        with_layout(df),
        base / "rm_loss_olmo.png",
        f"Layer-wise RegMean loss (Olmo-3-7b)",
        tick_every=N_SKIP,
    )

    # subsampled plot: keep every BLOCK_STRIDE-th transformer block (all 7
    # submodules within it) + lm_head -> far fewer x-points, label every point.
    BLOCK_STRIDE = 4
    df["block"] = df["layer"].map(block_of)
    blocks = sorted(b for b in df["block"].unique() if b < 10**6)
    keep = set(blocks[::BLOCK_STRIDE]) | {10**6}
    sub = df[df["block"].isin(keep)]
    line_plot(
        with_layout(sub),
        base / "rm_loss_olmo_subsampled.png",
        f"Layer-wise RegMean loss (Olmo-3-7b) [subsampled]",
        tick_every=1,
    )

    # --- total RegMean loss (summed over all layers) per method, bar plot ---
    import matplotlib.pyplot as plt

    methods = ["actmat", "regmean", "identity"]
    totals = df.groupby("method")["metric"].sum().reindex(methods)
    print("\ntotal RegMean loss (sum over all 225 layers):")
    for m in methods:
        print(f"  {m:9s} {totals[m]:.4g}")

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = sns.color_palette()[: len(methods)]
    bars = ax.bar(methods, totals.values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Loss")
    ax.set_title("Total RegMean loss across layers (Olmo-3-7b)")
    ax.bar_label(bars, fmt="%.3g", padding=3)
    fig.tight_layout()
    out_bar = REPO_ROOT / "artifacts/analysis/rm-loss/rm_loss_olmo_total.png"
    fig.savefig(out_bar, dpi=150, bbox_inches="tight")
    print(f"saved {out_bar}")


if __name__ == "__main__":
    main()
