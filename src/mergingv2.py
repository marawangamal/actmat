from argparse import Namespace
from pyexpat import model
import os
import sys
import time
from pathlib import Path
import torch
from typing import Callable, Optional, Sequence, Union
from tqdm import tqdm

from src.task_vectors import _TaskVector

pinv = torch.linalg.pinv

# ---------------------------------------------------------------------------
# Basic
# ---------------------------------------------------------------------------


def merge_sum(
    d: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # Shape: (N, Do, Di) -> (Do, Di)
    return d.sum(dim=0)


def merge_mean(
    d: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return d.mean(dim=0)


# ---------------------------------------------------------------------------
# TSV
# ---------------------------------------------------------------------------


def _compute_procrustes(x: torch.Tensor) -> torch.Tensor:
    u, _, vt = torch.linalg.svd(x, full_matrices=False)
    return u @ vt


def merge_tsv(d: torch.Tensor, **kwargs) -> torch.Tensor:
    """Computes the TSV merge of the given tensors.

    Computes: Uo  Dc Vto

    Args:
        tensors (torch.Tensor): The tensors to merge. Shape: (N_tasks, Di, Do)

    Returns:
        torch.Tensor: The merged tensors. Shape: (Di, Do)
    """

    N_tasks = len(d)
    u, s, vt = torch.linalg.svd(d, full_matrices=False)
    R = min(u.shape[1], vt.shape[2])
    Rp = R // N_tasks
    u, s, vt = u[:, :, :Rp], s[:, :Rp], vt[:, :Rp, :]

    # # # w/o decorrelation
    # tau_bl = torch.einsum("bij,bj,bjk->bik", u, s, vt)
    # tau[layer_name] = tau_bl.sum(dim=0)

    # w/ decorrelation
    B, Di, _ = u.shape
    _, _, Do = vt.shape
    # (Di, B, R)
    u_hat = u.permute(1, 0, 2).reshape(Di, B * Rp)
    s_hat = s.reshape(-1)
    vt_hat = vt.reshape(B * Rp, Do)
    u_ortho = _compute_procrustes(u_hat)  # (Di, Rp)
    vt_ortho = _compute_procrustes(vt_hat.T).T  # (Rp, Do)
    tau_l = torch.einsum("ij,j,jk->ik", u_ortho, s_hat, vt_ortho)  # (Di, Do)
    return tau_l


# ---------------------------------------------------------------------------
# ISOC
# ---------------------------------------------------------------------------
def merge_isoc(d: torch.Tensor, **kwargs):
    m = d.sum(dim=0)
    u, s, vt = torch.linalg.svd(m, full_matrices=False)
    s_iso = s.mean() * torch.ones_like(s)
    return torch.einsum("ik,k,kj->ij", u, s_iso, vt)


merge_isoc2 = lambda *args, **kwargs: merge_isoc(*args, **kwargs) * 2.0
merge_isoc3 = lambda *args, **kwargs: merge_isoc(*args, **kwargs) * 3.0


# ---------------------------------------------------------------------------
# RegMean
# ---------------------------------------------------------------------------


def merge_regmean(d: torch.Tensor, get_covariance: Callable, **kwargs):
    c = stat_fetcher_maps["covariance"]()
    return (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


# ---------------------------------------------------------------------------
# Fisher Merging
# ---------------------------------------------------------------------------
def _dinv(x, thresh=1e-8):
    return torch.where(x.abs() > thresh, 1 / x, 0)


def merge_fisher(
    d: torch.Tensor,
    stat_fetcher_maps: Callable,  # Shape: (N, Do*Di)
    **kwargs,
):
    f = stat_fetcher_maps["fisher"]()
    N, Do, Di = tau.shape
    return (_dinv(f.sum(dim=0)) * (f * d.reshape(N, Do * Di)).sum(dim=0)).reshape(
        Do, Di
    )


# ---------------------------------------------------------------------------
# ACTMat
# ---------------------------------------------------------------------------
def merge_actmat(d: torch.Tensor, **kwargs):
    c = d.transpose(1, 2) @ d
    return (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))
