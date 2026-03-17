#!/usr/bin/env python3
"""Compute L2 distances between merged param-folder models and finetuned models.

This compares models saved in the olmes param-folder format:
- param_manifest.json
- params/<tensor files>

For each (merged, finetuned) pair, the script computes:
    sqrt(sum_i ||theta_merged_i - theta_finetuned_i||_2^2)
across all parameters in the manifest, then prints a summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute L2 distances from merged param-folder models to finetuned param-folder models."
    )
    parser.add_argument(
        "--merged-dirs",
        "--merged-dir",
        nargs="+",
        dest="merged_dirs",
        required=True,
        help="One or more merged model param-folder directories.",
    )
    parser.add_argument(
        "--finetuned-dirs",
        nargs="+",
        required=True,
        help="One or more finetuned model param-folder directories to compare against.",
    )
    parser.add_argument(
        "--print-per-parameter",
        action="store_true",
        help="Also print each parameter's squared L2 contribution for every finetuned model.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to write the final merged-vs-finetuned distance table as CSV.",
    )
    return parser.parse_args()


def load_manifest(folder: Path) -> Dict:
    manifest_path = folder / "param_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = manifest.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"Manifest at {manifest_path} has empty or invalid 'params'.")
    return manifest


def build_param_file_path(model_dir: Path, manifest: Dict, param_name: str) -> Path:
    filename = manifest["params"][param_name]["file"]
    return model_dir / "params" / filename


def load_single_tensor(file_path: Path) -> torch.Tensor:
    import torch

    if file_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors_file
        except Exception as exc:
            raise RuntimeError(
                f"safetensors file found ({file_path}) but safetensors is not installed."
            ) from exc
        tensor_map = load_safetensors_file(str(file_path))
        if len(tensor_map) != 1:
            raise ValueError(f"Expected one tensor in {file_path}, found {len(tensor_map)}.")
        return next(iter(tensor_map.values()))
    return torch.load(file_path, map_location="cpu", weights_only=False)


def validate_same_keys(reference_manifest: Dict, other_manifest: Dict, other_dir: Path) -> Iterable[str]:
    reference_keys = list(reference_manifest["params"].keys())
    if set(reference_keys) != set(other_manifest["params"].keys()):
        diff = sorted(set(reference_keys).symmetric_difference(other_manifest["params"].keys()))
        preview = ", ".join(diff[:10])
        raise ValueError(
            f"Parameter keys differ for {other_dir}. First differing keys: {preview}"
        )
    return reference_keys


def squared_l2_between_tensors(
    merged_tensor: torch.Tensor, target_tensor: torch.Tensor
) -> torch.Tensor:
    import torch

    if merged_tensor.shape != target_tensor.shape:
        raise ValueError(
            f"Shape mismatch: merged has {tuple(merged_tensor.shape)}, "
            f"target has {tuple(target_tensor.shape)}"
        )

    # Accumulate in fp64 for stable totals across large checkpoints.
    merged_fp = merged_tensor.detach().to(dtype=torch.float64)
    target_fp = target_tensor.detach().to(dtype=torch.float64)
    diff = merged_fp - target_fp
    return torch.sum(diff * diff)


def extract_merge_method_label(model_dir: Path) -> str:
    name = model_dir.name.lower()
    tokens = [token for token in name.replace("_", "-").split("-") if token]

    if "regmean" in tokens:
        return "regmean"
    if "average" in tokens or "avg" in tokens:
        return "avg"
    if "eigcov" in tokens:
        return "eigcov"
    if "isoc" in tokens:
        return "isoc"
    if "tsv" in tokens:
        return "tsv"
    raise ValueError(
        f"Could not extract merged method name from directory: {model_dir}. "
        "Expected one of regmean, average/avg, eigcov, isoc, or tsv in the path name."
    )


def extract_model_label(model_dir: Path) -> str:
    return model_dir.name


def format_float(value: float) -> str:
    return f"{value:.6f}"


def print_distance_table(
    merged_labels: list[str],
    finetuned_labels: list[str],
    distance_matrix: list[list[float]],
) -> None:
    header = ["method"] + finetuned_labels
    rows = []
    for merged_label, row_values in zip(merged_labels, distance_matrix):
        rows.append([merged_label] + [format_float(value) for value in row_values])

    widths = []
    for col_idx in range(len(header)):
        column_values = [header[col_idx]] + [row[col_idx] for row in rows]
        widths.append(max(len(value) for value in column_values))

    def format_row(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))

    print("\nL2 Distance Table:")
    print(format_row(header))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def write_distance_table_csv(
    output_path: Path,
    merged_labels: list[str],
    finetuned_labels: list[str],
    distance_matrix: list[list[float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", *finetuned_labels])
        for merged_label, row_values in zip(merged_labels, distance_matrix):
            writer.writerow([merged_label, *row_values])


def compute_distance(
    merged_dir: Path,
    merged_manifest: Dict,
    target_dir: Path,
    target_manifest: Dict,
    print_per_parameter: bool,
) -> Tuple[float, int]:
    import torch
    from tqdm import tqdm

    param_names = validate_same_keys(merged_manifest, target_manifest, target_dir)

    total_squared_l2 = torch.zeros((), dtype=torch.float64)
    compared_params = 0

    progress_desc = f"{extract_merge_method_label(merged_dir)} vs {extract_model_label(target_dir)}"
    for param_name in tqdm(param_names, desc=progress_desc, unit="param"):
        merged_tensor = load_single_tensor(build_param_file_path(merged_dir, merged_manifest, param_name))
        target_tensor = load_single_tensor(build_param_file_path(target_dir, target_manifest, param_name))
        param_squared_l2 = squared_l2_between_tensors(merged_tensor, target_tensor)
        total_squared_l2 += param_squared_l2
        compared_params += 1

        if print_per_parameter:
            print(
                f"  {param_name}: squared_l2={param_squared_l2.item():.10e}"
            )

    return math.sqrt(total_squared_l2.item()), compared_params


def main() -> None:
    args = parse_args()

    merged_dirs = [Path(path).expanduser().resolve() for path in args.merged_dirs]
    finetuned_dirs = [Path(path).expanduser().resolve() for path in args.finetuned_dirs]
    output_csv = (
        Path(args.output_csv).expanduser().resolve() if args.output_csv is not None else None
    )

    merged_manifests = [(merged_dir, load_manifest(merged_dir)) for merged_dir in merged_dirs]
    target_manifests = [(target_dir, load_manifest(target_dir)) for target_dir in finetuned_dirs]

    print(f"Merged models: {len(merged_dirs)}")
    print(f"Finetuned models: {len(finetuned_dirs)}\n")

    merged_labels = [extract_merge_method_label(merged_dir) for merged_dir, _ in merged_manifests]
    finetuned_labels = [extract_model_label(target_dir) for target_dir, _ in target_manifests]
    distance_matrix = []

    for merged_dir, merged_manifest in merged_manifests:
        merged_label = extract_merge_method_label(merged_dir)
        print(f"Merged model: {merged_dir} ({merged_label})")
        row = []
        for target_dir, target_manifest in target_manifests:
            print(f"Comparing to: {target_dir}")
            distance, compared_params = compute_distance(
                merged_dir=merged_dir,
                merged_manifest=merged_manifest,
                target_dir=target_dir,
                target_manifest=target_manifest,
                print_per_parameter=args.print_per_parameter,
            )
            print(f"  parameters compared: {compared_params}")
            print(f"  l2_distance: {distance:.10f}\n")
            row.append(distance)
        distance_matrix.append(row)

    print_distance_table(
        merged_labels=merged_labels,
        finetuned_labels=finetuned_labels,
        distance_matrix=distance_matrix,
    )

    if output_csv is not None:
        write_distance_table_csv(
            output_path=output_csv,
            merged_labels=merged_labels,
            finetuned_labels=finetuned_labels,
            distance_matrix=distance_matrix,
        )
        print(f"\nSaved distance table to: {output_csv}")


if __name__ == "__main__":
    main()
