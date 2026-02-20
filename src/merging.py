from argparse import Namespace
from pyexpat import model
import sys
import time
from pathlib import Path
import torch
from typing import Callable, Sequence, Union
from tqdm import tqdm
import numpy as np

from src.task_vectors import _TaskVector


# Type: (key, [w1, w2, ...]) -> merged tensor
TensorMergeFn = Callable[[str, Sequence[torch.Tensor]], torch.Tensor]


def combine_task_vectors(
    vectors: Sequence[_TaskVector], merge: str, args: Namespace
) -> _TaskVector:
    """Generic combiner for task vectors.

    Args:
        vectors: list of task vectors (same logical type)
        merge: name of the function defined in this module to merge the task vectors

    Returns:
        A new task vector with the merged task vectors.
    """
    vectors = list(vectors)
    assert len(vectors) > 0, "Need at least one task vector"

    # Get the function (must be defined in this module)
    merge_fn = getattr(sys.modules[__name__], "merge_" + merge)

    # Prefer GPU for merging if available; results are moved back to CPU so they
    # stay compatible with the rest of the pipeline.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = vectors[0]
    # Cast others to same type as base
    casted = [base] + [base._cast_to_same_type(v) for v in vectors[1:]]

    start_time = time.time()

    with torch.no_grad():
        new_vector = {}
        for key in tqdm(
            casted[0].vector,
            desc="Merging task vectors",
            total=len(casted[0].vector),
            leave=False,
        ):
            if any(key not in v.vector for v in casted):
                # Skip keys that are not present in all vectors
                continue
            # Stack on the merge device
            # NOTE: use get_vector_element to speedup lazy mode with caching
            taus = torch.stack([v.get_vector_element(key).to(device) for v in casted])

            if (
                taus[0].ndim == 2
                and "text_projection" not in key
                and max(taus[0].shape) < 10_000
            ):
                # Only matrices can be merged using the merge function
                merged = merge_fn(taus, key=key, vectors=vectors)
            else:
                # For all other tensors, we average the values
                merged = taus.mean(dim=0)

            # Keep merged parameters on CPU for compatibility with checkpoint loading.
            # i.e. for when we do `task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)`
            new_vector[key] = merged.to("cpu")

    print(f"Merging task vectors took {(time.time() - start_time) / 60:.2f} minutes")
    return base.__class__(vector=new_vector)


def merge_sum(taus: torch.Tensor, **kwargs) -> torch.Tensor:
    # Shape: (N, Do, Di) -> (Do, Di)
    return taus.sum(dim=0)


def merge_mean(taus: torch.Tensor, **kwargs) -> torch.Tensor:
    return taus.mean(dim=0)


# ************************** ISOC **************************
def _merge_isoc(taus: torch.Tensor, mode="mean", **kwargs):
    m = taus.sum(dim=0)
    u, s, vt = torch.linalg.svd(m, full_matrices=False)
    if mode == "mean":
        s_iso = s.mean() * torch.ones_like(s)
    elif mode == "unity":
        s_iso = torch.ones_like(s)
    elif mode == "rms":
        s_iso = torch.sqrt((s**2).mean()) * torch.ones_like(s)
    elif mode == "spectral":
        s_iso = s[0] * torch.ones_like(s)  # Use largest singular value
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return torch.einsum("ik,k,kj->ij", u, s_iso, vt)


merge_isoc_mean = lambda *args, **kwargs: _merge_isoc(*args, mode="mean", **kwargs)
merge_isoc_rms = lambda *args, **kwargs: _merge_isoc(*args, mode="rms", **kwargs)


# ************************** KNOTS **************************
def _merge_knots(taus: torch.Tensor, merge_fn: Callable, **kwargs) -> torch.Tensor:
    # print device
    N, Do, Di = taus.shape
    d = taus.permute(1, 0, 2).reshape(Do, N * Di)
    u, s, vt = torch.linalg.svd(d, full_matrices=False)
    taus_tilde = torch.einsum("ij,jnk->nik", torch.diag(s), vt.reshape(-1, N, Di))
    tau_tilde = merge_fn(taus_tilde, **kwargs)
    return u @ tau_tilde


merge_knots_ta = lambda *args, **kwargs: _merge_knots(
    *args, merge_fn=merge_isoc_mean, **kwargs
)
merge_knots_isoc_mean = lambda *args, **kwargs: _merge_knots(
    *args, merge_fn=merge_isoc_rms, **kwargs
)
merge_knots_isoc_rms = lambda *args, **kwargs: _merge_knots(
    *args, merge_fn=merge_isoc_rms, **kwargs
)


# ************************** Subspace Alignment (SA) **************************
def merge_sa(taus: torch.Tensor, *args, **kwargs):
    m = torch.bmm(taus, taus.transpose(1, 2)).sum(dim=0)  # memory intensive
    # m = torch.einsum("nik,njk->ij", taus, taus)
    # check if nans/infs
    if torch.isnan(m).any() or torch.isinf(m).any():
        raise ValueError("m contains nans or infs")
    u, s, vt = torch.linalg.svd(m, full_matrices=False)
    q = u @ vt
    return (q @ taus).mean(dim=0)


pinv = torch.linalg.pinv


# ************************** RegMean **************************
def _param_key_to_module_key(key: str):
    return "image_encoder." + key.replace(".weight", "")


# Test normalized accuracy: 0.8934755825227096
# Test absolute accuracy: 0.6888413285902395
def merge_regmean(
    tau: torch.Tensor, key: str, vectors: Sequence[_TaskVector], **kwargs
):
    # Get project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    c = []
    km = _param_key_to_module_key(key)
    for v in vectors:
        task_name = v.metadata.get("task", None)
        model_name = v.metadata.get("model", None)
        if task_name is None:
            raise ValueError("No task name in metadata")
        if model_name is None:
            raise ValueError("No model name in metadata")
        cpath = project_root / "results" / model_name / f"covariance_{task_name}.npz"
        with np.load(cpath) as cdict:
            if km not in cdict:
                print(f"[skipped] {km} not found in {cpath}")
                return tau.mean(dim=0)
            c.append(cdict[km])
    c = torch.stack([torch.as_tensor(x, device=tau.device, dtype=tau.dtype) for x in c])
    # print(f"[success] {km} found in {cpath}")
    return (tau @ c).sum(dim=0) @ pinv(c.sum(dim=0))


# ************************** Eigenvalue Covariance (EigCov) **************************
def merge_eigcov(d: torch.Tensor, **kwargs):
    c = d.transpose(1, 2) @ d
    return (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))
