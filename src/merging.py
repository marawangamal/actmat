from argparse import Namespace
from pyexpat import model
import os
import sys
import time
from pathlib import Path
import torch
from typing import Callable, Optional, Sequence, Union
from tqdm import tqdm

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


def merge_regmean(d: torch.Tensor, **kwargs):
    stat_fetcher_maps = kwargs["stat_fetcher_maps"]
    c = []
    for fetchers in stat_fetcher_maps:
        ct = fetchers["covariance"]()
        if ct is None:
            return d.mean(dim=0)
        if not isinstance(ct, torch.Tensor):
            ct = torch.as_tensor(ct)
        c.append(ct)
    c = torch.stack([x.to(device=d.device, dtype=d.dtype) for x in c])
    return (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


def merge_regmean_w(d: torch.Tensor, w0: torch.Tensor, **kwargs):
    stat_fetcher_maps = kwargs["stat_fetcher_maps"]
    c = []
    for fetchers in stat_fetcher_maps:
        ct = fetchers["covariance"]()
        if ct is None:
            return d.mean(dim=0)
        if not isinstance(ct, torch.Tensor):
            ct = torch.as_tensor(ct)
        c.append(ct)
    c = torch.stack([x.to(device=d.device, dtype=d.dtype) for x in c])
    return ((d + w0) @ c).sum(dim=0) @ pinv(c.sum(dim=0)) - w0


def merge_regmean_w_fp64(d: torch.Tensor, w0: torch.Tensor, **kwargs):
    d = d.to(torch.float64)
    w0 = w0.to(torch.float64)
    stat_fetcher_maps = kwargs["stat_fetcher_maps"]
    c = []
    for fetchers in stat_fetcher_maps:
        ct = fetchers["covariance"]()
        if ct is None:
            return d.mean(dim=0)
        if not isinstance(ct, torch.Tensor):
            ct = torch.as_tensor(ct)
        c.append(ct)
    c = torch.stack([x.to(device=d.device, dtype=d.dtype) for x in c])
    return ((d + w0) @ c).sum(dim=0) @ pinv(c.sum(dim=0)) - w0


def _interp_cov(c: torch.Tensor, angular_distance: float) -> torch.Tensor:
    """Point on the segment c -> I at `angular_distance` (radians) from c.

    The angle (Frobenius inner product) between c and (1-s)c + sI is monotone
    in s, with closed-form inverse; s is clamped to [0, 1] so angles beyond
    angle(c, I) saturate at the identity.
    """
    eye = torch.eye(c.shape[-1], device=c.device, dtype=c.dtype)
    na = torch.linalg.norm(c)
    p = torch.trace(c) / na  # component of I along c
    q = torch.linalg.norm(eye - (p / na) * c)  # component of I orthogonal to c
    if q == 0:  # c is a multiple of I; every point on the segment has angle 0
        return c
    theta = torch.as_tensor(angular_distance, device=c.device, dtype=c.dtype)
    theta_max = torch.atan2(q, p)  # angle(c, I), reached at s=1
    t = torch.tan(theta.clamp(min=0.0).minimum(theta_max))
    s = (na * t / (q + (na - p) * t)).clamp(0.0, 1.0)
    return (1 - s) * c + s * eye


def merge_regmean_interp(d: torch.Tensor, **kwargs):
    """RegMean with each covariance moved toward I by `angular_distance`.

    `angular_distance` is in units of pi (0 = aligned, 0.5 = orthogonal),
    matching scripts/vision/generate_error_terms.py.
    """
    angular_distance = kwargs.get("angular_distance", 0.0) * torch.pi
    stat_fetcher_maps = kwargs["stat_fetcher_maps"]
    c = []
    for fetchers in stat_fetcher_maps:
        ct = fetchers["covariance"]()
        if ct is None:
            return d.mean(dim=0)
        if not isinstance(ct, torch.Tensor):
            ct = torch.as_tensor(ct)
        c.append(ct)
    c = torch.stack(
        [_interp_cov(x.to(device=d.device, dtype=d.dtype), angular_distance) for x in c]
    )
    return (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


# ---------------------------------------------------------------------------
# Fisher Merging
# ---------------------------------------------------------------------------
def _dinv(x, thresh=1e-8):
    return torch.where(x.abs() > thresh, 1 / x, 0)


def merge_fisher(
    d: torch.Tensor,
    stat_fetcher_maps: Sequence[dict],  # Shape: (N, Do*Di)
    **kwargs,
):
    N, Do, Di = d.shape
    f = []
    for fetchers in stat_fetcher_maps:
        ft = fetchers["fisher"]()
        if ft is None:
            return d.mean(dim=0)
        if not isinstance(ft, torch.Tensor):
            ft = torch.as_tensor(ft)
        f.append(ft)
    f = torch.stack([x.reshape(-1).to(device=d.device, dtype=d.dtype) for x in f])
    return (_dinv(f.sum(dim=0)) * (f * d.reshape(N, Do * Di)).sum(dim=0)).reshape(
        Do, Di
    )


# ---------------------------------------------------------------------------
# ACTMat
# ---------------------------------------------------------------------------
def merge_actmat(d: torch.Tensor, **kwargs):
    c = d.transpose(1, 2) @ d
    return (d @ c).sum(dim=0) @ pinv(c.sum(dim=0))


def merge_actmat_w(d: torch.Tensor, w0: torch.Tensor, **kwargs):
    c = d.transpose(1, 2) @ d
    return ((d + w0) @ c).sum(dim=0) @ pinv(c.sum(dim=0)) - w0


def merge_actmat_w_fp64(d: torch.Tensor, w0: torch.Tensor, **kwargs):
    d = d.to(torch.float64)
    w0 = w0.to(torch.float64)
    c = d.transpose(1, 2) @ d
    return ((d + w0) @ c).sum(dim=0) @ pinv(c.sum(dim=0)) - w0


def merge_actmat_herm(d: torch.Tensor, **kwargs):
    c = d.transpose(1, 2) @ d
    return (d @ c).sum(dim=0) @ pinv(c.sum(dim=0), hermitian=True)


def merge_actmat_herm_10ki(d: torch.Tensor, **kwargs):
    if d.shape[-1] > 10_000:
        return d.mean(dim=0)
    return merge_actmat_herm(d)


def merge_actmat_gd(
    d: torch.Tensor,
    lam=0.0,
    alpha_weighted=False,
    cov_weighted=False,
    lr=1e-5,
    max_iters=300,
    thresh=-float("inf"),
    **kwargs,
) -> torch.Tensor:
    C = d.transpose(1, 2) @ d

    if cov_weighted:
        C = C / (torch.linalg.norm(C, ord="fro", dim=(-2, -1), keepdim=True) ** 2)

    if alpha_weighted:
        alpha = 1.0 / d.flatten(1).norm(dim=1)
        C = alpha[:, None, None] * C

    W = d.mean(dim=0).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([W], lr=lr, weight_decay=0.0)

    with torch.enable_grad():
        prev_loss = float("inf")
        pbar = tqdm(range(int(max_iters)), desc="Gradient descent", leave=False)
        for i in pbar:
            optimizer.zero_grad()
            diff = W.unsqueeze(0) - d
            loss = (diff @ C).mul_(diff).sum()
            if lam > 0:
                loss = loss + lam * W.square().sum()
            loss.backward()
            optimizer.step()

            cur_loss = loss.item()
            if abs(prev_loss - cur_loss) / (abs(prev_loss) + 1e-12) < thresh:
                print(f"[converged] loss={cur_loss:.1e} < {thresh:.1e}")
                break
            prev_loss = cur_loss
            pbar.set_postfix(loss=cur_loss)

    if i == int(max_iters) - 1:
        print(f"[not converged] loss={cur_loss:.1e} after {max_iters} iters")

    return W.detach()


def merge_actmat_gd_10ki(d: torch.Tensor, **kwargs):
    if d.shape[-1] > 10_000:
        return d.mean(dim=0)
    return merge_actmat_gd(d, **kwargs)


# ---------------------------------------------------------------------------
# WUDI (Li et al., ICML 2025 — https://arxiv.org/abs/2503.08099)
# ---------------------------------------------------------------------------
def merge_wudi(
    d: torch.Tensor,
    wudi_iters: int = 300,
    wudi_lr: float = 1e-5,
    wudi_weighted: bool = True,
    **kwargs,
) -> torch.Tensor:
    """WUDI: data-free merging by minimizing per-task interference.

    Optimizes M (init: Σ_i τ_i) for `wudi_iters` Adam steps at `wudi_lr` to
    minimize Σ_i ‖(M − τ_i) τ_iᵀ‖_F² / ‖τ_i‖_F².

    When `wudi_weighted=False`, drops the 1/‖τ_i‖² normalization — tasks with
    larger task-vector norm then dominate the objective (`merge_wudi_unweighted`).
    """
    N = d.shape[0]
    d_det = d.detach()
    if wudi_weighted:
        l2_sq = d_det.reshape(N, -1).pow(2).sum(dim=-1).view(N, 1, 1).clamp_min(1e-12)
    with torch.enable_grad():
        M = torch.nn.Parameter(d_det.sum(dim=0).clone())
        optimizer = torch.optim.Adam([M], lr=wudi_lr, weight_decay=0.0)
        for _ in tqdm(range(wudi_iters), desc="WUDI", leave=False):
            disturbing = M.unsqueeze(0) - d_det
            inner = torch.matmul(disturbing, d_det.transpose(1, 2))
            sq = inner.pow(2)
            loss = (sq / l2_sq).sum() if wudi_weighted else sq.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return M.detach()
